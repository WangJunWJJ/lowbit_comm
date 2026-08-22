#include "dequant_api.cuh"

#include "utils.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__global__ void qwd_refresh_cast_kernel(
    const float* gathered,
    __half* output,
    int64_t numel) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; index < numel; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = __float2half(gathered[index]);
  }
}

}  // namespace

bool try_inplace_dequantize_gathered_add(
    const torch::Tensor& input,
    torch::Tensor& output,
    int64_t world_size,
    int64_t payload_bytes_per_rank,
    int64_t shard_numel,
    int64_t original_numel) {
  return inplace_dequantize_gathered_add(
      input,
      output,
      64,
      0,
      8,
      QuantType::Linear,
      true,
      world_size,
      payload_bytes_per_rank,
      payload_bytes_per_rank,
      shard_numel,
      original_numel);
}

bool try_inplace_qwd_refresh_cast(
    const torch::Tensor& gathered,
    torch::Tensor& output) {
  if (gathered.scalar_type() != at::kFloat ||
      output.scalar_type() != at::kHalf) {
    return false;
  }
  TORCH_CHECK(gathered.is_cuda(), "gathered master must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "qWD output must be a CUDA tensor");
  TORCH_CHECK(gathered.is_contiguous(), "gathered master must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "qWD output must be contiguous");
  TORCH_CHECK(
      gathered.device() == output.device(),
      "gathered master and qWD output must be on the same device");
  TORCH_CHECK(
      gathered.numel() == output.numel(),
      "gathered master and qWD output must have the same numel");
  if (gathered.numel() == 0) return true;

  c10::cuda::CUDAGuard device_guard(output.device());
  int64_t blocks = (output.numel() + kThreads - 1) / kThreads;
  blocks = std::min<int64_t>(blocks, 65535);
  qwd_refresh_cast_kernel<<<
      static_cast<int>(blocks),
      kThreads,
      0,
      get_current_cuda_stream()>>>(
      static_cast<const float*>(gathered.data_ptr()),
      static_cast<__half*>(output.data_ptr()),
      output.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}
