# BrushGS
BrushGS is a simple and lightweight library for creating a brush effect on the canvas.

## Installation


## Requirements


## Running


## Simplified Usage
Train Related:
1. SPARSE_ADAM_AVAILABLE, remove for training with Adam optimizer.
2. OptimizationParams, since not used in rendering.
3. In gaussian_model.py, remove the following: distCUDA2, create_from_random/pcd, oneupSHdegree, restore, training_setup, capture, ...

Miscs:
1. separate_sh, compute SH --> RGB in CUDA
2. compute_cov3D_python, compute cov3D in CUDA
2. skip_train & skip_test, default to render all images
3. train_test_exp, remove for rendering