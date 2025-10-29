# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

# Additional modifications made by Zhizheng Xiang, 2025
# distributed under the same terms for non-commercial purposes.

import os
from os import makedirs
import math
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision

from scene import Scene
from utils.general_utils import safe_state, load_image
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from brush_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           scaling_modifier = 1.0, texture = None, shading = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get the means, opacities, scales, rotations, and SHs of the Gaussians
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        shs = shs,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        texture = texture,
        normal = shading
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, texture, shading):
    render_path = os.path.join(model_path, name, "cluster_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "cluster_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, 1.0, texture, shading)["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, pipeline: PipelineParams, num_cluster: int, texture_path: str):
    with torch.no_grad():
        if texture_path:
            texture, shading = load_image(texture_path)
        else:
            texture = torch.Tensor([])
            shading = torch.Tensor([])

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=num_cluster, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Render training and test sets
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                   gaussians, pipeline, background, texture, shading)

        # render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
        #            gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Renderer parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("-n", "--num_cluster", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-t", "--texture", default='', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args.num_cluster, args.texture)