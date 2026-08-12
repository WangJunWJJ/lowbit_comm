#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

cudaStream_t get_current_cuda_stream() {
    auto stream = at::cuda::getCurrentCUDAStream();
    return stream.stream();
}