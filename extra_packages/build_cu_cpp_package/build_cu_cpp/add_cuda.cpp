#include <torch/extension.h>

// Implemented in add_kernel.cu
torch::Tensor add_cuda_forward(torch::Tensor a, torch::Tensor b);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return add_cuda_forward(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Simple elementwise add (CUDA, float32)");
}
