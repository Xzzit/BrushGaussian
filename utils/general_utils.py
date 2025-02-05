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
    img = Image.open(path)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor() # [0, 1] with shape (C, H, W)
    ])
    tensor = transform(img)

    if img.mode == "L":
        print("L mode detected, using grayscale as alpha channel.")
    elif img.mode == "LA":
        print("LA mode detected, using alpha channel.")
        tensor = tensor[1].unsqueeze(0).to("cuda")
    elif img.mode == "RGB":
        print("Warning: RGB mode detected. Please provide an alpha channel.")
        tensor = torch.Tensor([])
    elif img.mode == "RGBA":
        print("RGBA mode detected, using alpha channel.")
        tensor = tensor[3].unsqueeze(0).to("cuda")

    return tensor

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
