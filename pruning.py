import os
import shutil
import torch
from sklearn.cluster import MiniBatchKMeans
from argparse import ArgumentParser, Namespace

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import SH2RGB


def main():
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str) # Pretrained 3DGS directory
    parser.add_argument('-i', '--iteration', type=int, default=-1) # Iteration to prune
    parser.add_argument('-n', '--num_clusters', type=int, default=-1) # Number of clusters to prune to
    args = parser.parse_args()

    # Load info
    cfgfilepath = os.path.join(args.model, "cfg_args")
    try:
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
            args_cfgfile = eval(cfgfile_string)
            cfg_dict = vars(args_cfgfile).copy()
    except FileNotFoundError:
        print("Invalid model path.")

    # Load Gaussians
    gaussians = GaussianModel(sh_degree=cfg_dict['sh_degree'])
    if args.iteration == -1:
        args.iteration = searchForMaxIteration(os.path.join(args.model, "point_cloud"))
    gaussians.load_ply(os.path.join(args.model, "point_cloud",
                                    f"iteration_{args.iteration}",
                                    "point_cloud.ply"))
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians from iteration {args.iteration}")

    # 1st. Remove Gaussians with low I
    # I = opacity × trace(cov) × luminance
    means3D = gaussians.get_xyz
    opacities = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features
    L = build_scaling_rotation(scales, rotations)
    sigma3 = L @ L.transpose(1, 2) # 3D Covariance matrix (Nx3x3)
    trace3 = sigma3.diagonal(dim1=1, dim2=2).sum(-1) # 3D Covariance trace (N)
    rgb3 = torch.clamp(SH2RGB(shs[:, 0, :]), min=0, max=1)
    lumi = 0.2126 * rgb3[:, 0] + 0.7152 * rgb3[:, 1] + 0.0722 * rgb3[:, 2] # RGB to luminance
    I = opacities * trace3.unsqueeze(1) * lumi.unsqueeze(1)
    _, idx = I.squeeze().sort(descending=False) # [N]
    cutoff = int(gaussians.get_xyz.shape[0] * 0.3)
    gaussians._xyz = means3D[idx[cutoff:]].clone().detach()
    gaussians._opacity = gaussians.inverse_opacity_activation(opacities[idx[cutoff:]].clone().detach())
    gaussians._scaling = gaussians.scaling_inverse_activation(scales[idx[cutoff:]].clone().detach())
    gaussians._rotation = rotations[idx[cutoff:]].clone().detach()
    gaussians._features_dc = shs[idx[cutoff:], 0:1, :].contiguous().clone().detach()
    gaussians._features_rest = shs[idx[cutoff:], 1:, :].contiguous().clone().detach()
    print(f"Removed {cutoff} Gaussians with low opacity and luminance. Remaining: {gaussians.get_xyz.shape[0]}")

    # 2nd. KMeans clustering based on xyz and RGB
    rgb3 = torch.clamp(SH2RGB(gaussians._features_dc[:, 0, :]), min=0, max=1)
    fv = torch.cat([gaussians.get_xyz, gaussians.get_scaling, rgb3], dim=-1)
    if args.num_clusters == -1:
        # num_clusters = int(gaussians.get_xyz.shape[0] * 0.01)
        num_clusters = 7000
    else:
        num_clusters = args.num_clusters
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=8192)
    cluster_labels = kmeans.fit_predict(fv.cpu().numpy())
    num_points_list = []

    means3D = torch.empty((num_clusters, 3), device="cuda")
    opacities = torch.empty((num_clusters, 1), device="cuda")
    scales = torch.empty((num_clusters, 3), device="cuda")
    rotations = torch.empty((num_clusters, 4), device="cuda")
    shs = torch.empty((num_clusters, 16, 3), device="cuda")

    for i in range(num_clusters):
        mask = (cluster_labels == i)
        num_points = gaussians.get_xyz[mask].shape[0]
        num_points_list.append(num_points)
        means3D[i] = torch.mean(gaussians.get_xyz[mask], dim=0)
        opacities[i] = torch.mean(gaussians.get_opacity[mask], dim=0)
        scales[i] = torch.mean(gaussians.get_scaling[mask], dim=0) * int(num_points ** (1/4))
        rotations[i] = torch.mean(gaussians.get_rotation[mask], dim=0)
        shs[i] = torch.mean(gaussians.get_features[mask], dim=0)
    print(f"After clustering. Remaining: {num_clusters}")

    # 3rd. Remove floating Gaussians
    num_points_list = torch.tensor(num_points_list)
    if num_clusters > 1000:
        k = torch.argsort(num_points_list)[:int(num_clusters * 0.1)]
        k_opa = torch.argsort(opacities[:, 0])[:int(num_clusters * 0.1)]
        mask = torch.ones(num_clusters, dtype=torch.bool)
        mask[k] = False
        mask[k_opa] = False
        means3D = means3D[mask]
        opacities = opacities[mask]
        opacities = (opacities - opacities.min()) / (opacities.max() - opacities.min()) * 0.3 + 0.7
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

    # Set the values
    gaussians._xyz = means3D.clone().detach()
    gaussians._opacity = gaussians.inverse_opacity_activation(opacities.clone().detach())
    gaussians._scaling = gaussians.scaling_inverse_activation(scales.clone().detach())
    gaussians._rotation = rotations.clone().detach()
    gaussians._features_dc = shs[:, 0:1, :].contiguous().clone().detach()
    gaussians._features_rest = shs[:, 1:, :].contiguous().clone().detach()
    print(f"After removing floating GS. Remaining: {gaussians.get_xyz.shape[0]}")

    # Save the Gaussians
    dst_dir = os.path.join("output", args.model.split(os.sep)[-1])
    gaussians.save_ply(os.path.join(dst_dir, "point_cloud", f"cluster_{num_clusters}", "point_cloud.ply"))
    shutil.copy(cfgfilepath, os.path.join(dst_dir, "cfg_args"))
    
if __name__ == "__main__":
    main()