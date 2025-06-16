# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

# Additional modifications made by Zhizheng Xiang, 2025
# distributed under the same terms for non-commercial purposes.

import sys
import random
from datetime import datetime
from PIL import Image

import torch
import torchvision
from scipy.ndimage import sobel, gaussian_filter
import numpy as np

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def load_image(path, size=256):
    img = Image.open(path).convert('RGBA')

    img_np = np.array(img)
    alpha = img_np[:, :, 3]
    non_zero_indices = np.argwhere(alpha > 0)

    if non_zero_indices.size == 0:
        print("Warning: Whole image alpha == 0. Returning empty tensor.")
        return torch.Tensor([])

    top_left = non_zero_indices.min(axis=0)
    bottom_right = non_zero_indices.max(axis=0)

    top, left = top_left
    bottom, right = bottom_right

    cropped_img = img.crop((left, top, right+1, bottom+1))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor() # [0, 1] with shape (C, H, W)
    ])
    tensor_img = transform(cropped_img)
    tensor_img = tensor_img[3].unsqueeze(0).to("cuda")

    # Compute lighting normals
    img_rgb = np.asarray(cropped_img).astype(np.float32) / 255.0
    img_gray = 0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1] + 0.1140 * img_rgb[..., 2]

    height_map = gaussian_filter(img_gray, sigma=1)

    dx = sobel(height_map, axis=1)  # Sobel in X direction
    dy = sobel(height_map, axis=0)  # Sobel in Y direction
    dz = np.ones_like(height_map) * 0.5

    normals = np.stack((-dx, -dy, dz), axis=-1)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= np.maximum(norms, 1e-8)

    # Light direction
    light_dir = np.array([1, 1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Step 4: Compute diffuse shading
    diffuse = np.clip(np.sum(normals * light_dir, axis=2), 0, 1)
    ambient = 0.5
    shading = np.clip(ambient + diffuse * 0.7, 0, 1)
    shading = shading.astype(np.float32)

    tensor_shading = torch.from_numpy(shading).unsqueeze(0).to("cuda")
    return tensor_img, tensor_shading

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    S = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    S[:,0,0] = s[:,0]
    S[:,1,1] = s[:,1]
    S[:,2,2] = s[:,2]

    L = R @ S
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))