# BrushGaussian: Brushstroke-Based Stylization for 3D Gaussian Splatting

<p align="center">
  <img src="assets/fig1.png" alt="description" width="720"/>
</p>

Abstract: We present a method for enhancing 3D Gaussian Splatting primitives with brushstroke-aware stylization. Previous approaches to 3D style transfer are typically limited to color or texture modifications, lacking an understanding of artistic shape deformation. In contrast, we focus on individual 3D Gaussian primitives, exploring their potential to enable style transfer that incorporates both color- and brushstroke-inspired local geometric stylization. Specifically, we introduce additional texture features for each Gaussian primitive and apply a texture mapping technique to achieve brushstroke-like geometric effects in a rendered scene. Furthermore, we propose an unsupervised clustering algorithm to efficiently prune redundant Gaussians, ensuring that our method seamlessly integrates with existing 3D Gaussian Splatting pipelines. Extensive evaluations demonstrate that our approach outperforms existing baselines by producing brushstroke-aware artistic renderings with richer geometric expressiveness and enhanced visual appeal.

## Installation
if you have [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) installed, just run
```bash
scikit-learn=1.0.2
```
otherwise, you can create a new conda environment by running:
```bash
conda env create -f environment.yml
```

## Running
### Step 0: (Optional) Train a Gaussian Splatting Model
If you don't already have a pre-trained model, follow the instructions in the [3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) to train a Gaussian Splatting model on your dataset.

### Step 1: Pruning and Clustering
Run the following command to perform pruning and clustering:
```bash
python pruning.py -m <path_to_your_model> -n <number_of_clusters>
```

### Step 2: Stylized Rendering
Render the stylized Gaussian model using a style image:
```bash
python render.py -m <path_to_your_model> --texture <style_image_path>
```