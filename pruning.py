import os
import torch
from sklearn.cluster import MiniBatchKMeans
from argparse import ArgumentParser, Namespace

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import SH2RGB


def get_gs_features(gaussians: GaussianModel):
    means3D = gaussians.get_xyz
    opacities = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features
    return means3D, opacities, scales, rotations, shs

def main():
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-i', '--iteration', type=int, default=-1)
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
    [means3D, opacities, scales, rotations, shs] = get_gs_features(gaussians)
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
    num_clusters = 8000
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=8192)
    cluster_labels = kmeans.fit_predict(fv.cpu().numpy())
    num_points_list = []

    means3D = torch.empty((num_clusters, 3), device="cuda")
    opacities = torch.ones((num_clusters, 1), device="cuda")
    scales = torch.empty((num_clusters, 3), device="cuda")
    rotations = torch.empty((num_clusters, 4), device="cuda")
    shs = torch.empty((num_clusters, 16, 3), device="cuda")

    for i in range(num_clusters):
        mask = (cluster_labels == i)
        num_points = gaussians.get_xyz[mask].shape[0]
        num_points_list.append(num_points)
        means3D[i] = torch.mean(gaussians.get_xyz[mask], dim=0)
        opacities[i] *= (torch.rand(1, device="cuda") * 0.7 + 0.3)
        scales[i] = torch.mean(gaussians.get_scaling[mask], dim=0) * int(num_points ** (1/3))
        rotations[i] = torch.mean(gaussians.get_rotation[mask], dim=0)
        shs[i] = torch.mean(gaussians.get_features[mask], dim=0)

    # 3rd. Remove floating Gaussians
    num_points_list = torch.tensor(num_points_list)
    if num_clusters > 1000:
        k = torch.argsort(num_points_list)[:int(num_clusters * 0.05)] # 5% of the clusters to remove (floating Gaussians)
        mask = torch.ones(num_clusters, dtype=torch.bool)
        mask[k] = False
        means3D = means3D[mask]
        opacities = opacities[mask]
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
    print(f"Clustering completed. Remaining: {gaussians.get_xyz.shape[0]}")

    gaussians.save_ply(os.path.join(args.model, "point_cloud", "iteration_37",
                                    "point_cloud.ply"))
    

if __name__ == "__main__":
    main()