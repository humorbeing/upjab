#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void add_forward_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void add_backward_kernel(const float* __restrict__ grad_out,
                                    float* __restrict__ grad_a,
                                    float* __restrict__ grad_b,
                                    int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // d(a+b)/da = 1, d(a+b)/db = 1
        float g = grad_out[idx];
        grad_a[idx] = g;
        grad_b[idx] = g;
    }
}

torch::Tensor add_forward_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto out = torch::empty_like(a);
    const auto n = a.numel();

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    add_forward_kernel<<<blocks, threads, 0, 0>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA forward kernel launch failed");
    return out;
}

std::vector<torch::Tensor> add_backward_cuda(torch::Tensor grad_out) {
    CHECK_INPUT(grad_out);
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be float32");

    auto grad_a = torch::empty_like(grad_out);
    auto grad_b = torch::empty_like(grad_out);
    const auto n = grad_out.numel();

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    add_backward_kernel<<<blocks, threads, 0, 0>>>(
        grad_out.data_ptr<float>(),
        grad_a.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        n
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA backward kernel launch failed");
    return {grad_a, grad_b};
}
