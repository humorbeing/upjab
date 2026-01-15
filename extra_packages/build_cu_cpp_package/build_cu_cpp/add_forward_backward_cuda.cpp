#include <torch/extension.h>
#include <vector>

// Implemented in add_kernel.cu
torch::Tensor add_forward_cuda(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> add_backward_cuda(torch::Tensor grad_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_forward_cuda, "Add forward (CUDA, float32)");
    m.def("backward", &add_backward_cuda, "Add backward (CUDA, float32)");
}
