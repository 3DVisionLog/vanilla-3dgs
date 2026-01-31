#include "rasterizer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 파이썬에서 함수 이름 뭘로 부를지
    m.def("emitGaussianTilePairs", &emitGaussianTilePairs, "Preprocess Gaussian tiling");
    m.def("forwardCUDA", &forwardCUDA, "Render Forward (CUDA)");
    m.def("backwardCUDA", &backwardCUDA, "Render Backward (CUDA)");
}