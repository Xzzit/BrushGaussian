# BrushGaussian: Brushstroke-Based Stylization for 3D Gaussian Splatting

<p align="center">
  <img src="assets/fig1.png" alt="description" width="720"/>
</p>

BrushGaussian is a method for enhancing 3D Gaussian Splatting with brushstroke-aware stylization — supporting both color transfer and artistic shape deformation. Unlike previous 3D style transfer methods that only adjust color or texture, we enable fine-grained control over geometry, inspired by real brushstrokes.

✨ Our method:

* Adds per-Gaussian texture features to simulate brushstroke-like effects;

* Applies a custom texture mapping technique for geometric stylization;

* Uses unsupervised clustering to prune redundant Gaussians for efficiency;

* Is fully compatible with existing 3DGS pipelines.

## Cloning
```bash
git clone https://github.com/Xzzit/BrushGS --recursive
cd BrushGS
```

## Installation
There are two ways to set up the environment depending on whether you've already installed [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

### Option 1: You already have 3DGS installed
```bash
conda activate <your_3DGS_env>
pip install scikit-learn==1.0.2
pip install brush-gaussian-rasterization\.
```

### Option 2: You don't have 3DGS installed
Create a new conda environment with all required packages:
```bash
conda env create -f environment.yml
```
💡Tip: The default environment name is brushGS, but feel free to change it in environment.yml if needed.

## Running
### Step 0: (Optional) Prepare a 3D Gaussian Splatting Model
If you don't already have a pre-trained 3DGS model, follow the instructions in the [3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) to train a Gaussian Splatting model on your dataset.

After training, the output directory of your model should look like this:
```
name_of_the_scene/
├── point_cloud/               # Contains Gaussian checkpoints
│   ├── iteration_7000/
│   │   └── point_cloud.ply
│   ├── iteration_30000/
│   └── ...
├── cameras.json
├── exposure.json
├── cfg_args
├── input.ply
├── train/                     # Optional
├── test/                      # Optional
```

### Step 1: Pruning and Clustering
Run the following command to prune the original Gaussians and perform clustering:
```bash
python pruning.py -m <path_to_your_model_name> -n <number_of_clusters>
```

Example:
```bash
python pruning.py -m E:\gaussian-splatting\output\mic -n 250
```

Arguments:
`-m <path_to_your_model_name>`: Path to the trained 3DGS model directory.
`-n <number_of_clusters>`: Number of output clusters (i.e., the number of retained Gaussian primitives).
This value depends on the complexity of your scene:
* 250 ~ 1000 for a simple object scene (e.g., NeRF synthetic)
* 7000 ~ for complex scenes (e.g., Tanks and Temples)

### Step 2: Stylized Rendering
Render the stylized Gaussian model using a style image:
```bash
python render.py -m <path_to_your_model> -t <style_image_path>
```

Example:
```bash
python render.py -m output\mic -t texture\splatting.png
```

Arguments:
`-m <path_to_your_model>`: Path to the pruned Gaussian model directory.
`-t <style_image_path>`: Path to the style image you want to apply. For now, we only support RGBA images with alpha channels. If not provided, a elliptical rendering will be performed.
`-n <number_of_cluster>`: Which cluster to be rendered. Default is -1, stands for max cluster.