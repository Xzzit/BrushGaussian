/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Additional modifications made by Zhizheng Xiang, 2025
 * distributed under the same terms for non-commercial purposes.
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("brush_gaussian", &RasterizeGaussiansCUDA);
  m.def("mark_visible", &markVisible);
}