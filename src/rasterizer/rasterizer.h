#pragma once
#include <torch/extension.h> // 파이썬과 C++을 이어주는 PyTorch C++ API 헤더

std::vector<torch::Tensor> emitGaussianTilePairs(
    torch::Tensor means2d,
    torch::Tensor radii,
    int H,
    int W
);

torch::Tensor forwardCUDA(
    torch::Tensor means2d,
    torch::Tensor conics,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor radii,
    torch::Tensor tile_ranges,
    int H,
    int W
);

std::vector<torch::Tensor> backwardCUDA(
    torch::Tensor means2d,
    torch::Tensor conics,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor radii,
    torch::Tensor tile_ranges,
    torch::Tensor final_image,
    torch::Tensor grad_image
);
