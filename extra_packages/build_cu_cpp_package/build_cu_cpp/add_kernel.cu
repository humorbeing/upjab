#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda_forward(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto out = torch::empty_like(a);
    const auto n = a.numel();

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    // Launch on the default CUDA stream (stream 0)
    add_kernel<<<blocks, threads, 0, 0>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return out;
}
