#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

#include "metadata_api.cuh"

namespace {

constexpr int64_t kPacketWords = 24;
constexpr int64_t kDescriptorWords = 12;
constexpr int64_t kMaximumDimensions = 8;
constexpr int64_t kDimensionsOffset = 16;

__global__ void decode_dynamic_metadata_kernel(
    const int64_t* metadata,
    int64_t* descriptors,
    int64_t world_size,
    int64_t dtype_code,
    int64_t bit,
    int64_t group_size,
    int64_t quant_type_code,
    int64_t compact,
    int64_t layout_generation,
    int64_t max_numel,
    int64_t payload_stride
) {
    const int64_t rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= world_size) return;
    const int64_t* packet = metadata + rank * kPacketWords;
    int64_t* descriptor = descriptors + rank * kDescriptorWords;
    int64_t status = 0;
    const int64_t ndim = packet[1];
    if (packet[0] != 1) status = 1;
    else if (ndim < 0 || ndim > kMaximumDimensions) status = 2;
    else if (packet[2] != dtype_code) status = 3;
    else if (packet[4] != bit || packet[5] != group_size) status = 4;
    else if (packet[6] != quant_type_code || packet[7] != compact) status = 5;
    else if (packet[8] != layout_generation) status = 6;
    else if (packet[3] < 0 || packet[3] > payload_stride) status = 7;
    for (int64_t index = 11; index < kDimensionsOffset && status == 0; ++index) {
        if (packet[index] != 0) status = 8;
    }
    int64_t logical_numel = 1;
    for (int64_t index = 0; index < kMaximumDimensions; ++index) {
        const int64_t dimension = packet[kDimensionsOffset + index];
        descriptor[4 + index] = dimension;
        if (status != 0) continue;
        if (index >= ndim) {
            if (dimension != 0) status = 9;
            continue;
        }
        if (dimension < 0) {
            status = 10;
        } else if (dimension == 0) {
            logical_numel = 0;
        } else if (logical_numel > max_numel / dimension) {
            status = 11;
        } else {
            logical_numel *= dimension;
        }
    }
    if (status == 0 && (logical_numel != packet[10] || logical_numel > max_numel)) {
        status = 12;
    }
    const int64_t groups = (logical_numel + group_size - 1) / group_size;
    const int64_t scale_bytes = dtype_code == 3 ? 4 : 2;
    const int64_t expected_payload = groups * (group_size * bit / 8 + scale_bytes);
    if (status == 0 && packet[3] != expected_payload) status = 13;
    descriptor[0] = status;
    descriptor[1] = ndim;
    descriptor[2] = packet[3];
    descriptor[3] = logical_numel;
}

}  // namespace

void inplace_decode_dynamic_metadata(
    torch::Tensor metadata,
    torch::Tensor descriptors,
    int64_t world_size,
    int64_t dtype_code,
    int64_t bit,
    int64_t group_size,
    int64_t quant_type_code,
    bool compact,
    int64_t layout_generation,
    int64_t max_numel,
    int64_t payload_stride
) {
    TORCH_CHECK(metadata.is_cuda() && descriptors.is_cuda(), "metadata tensors must be CUDA");
    TORCH_CHECK(metadata.scalar_type() == torch::kInt64, "metadata must be int64");
    TORCH_CHECK(descriptors.scalar_type() == torch::kInt64, "descriptors must be int64");
    TORCH_CHECK(metadata.is_contiguous() && descriptors.is_contiguous(), "metadata tensors must be contiguous");
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    TORCH_CHECK(metadata.numel() == world_size * kPacketWords, "metadata packet size mismatch");
    TORCH_CHECK(descriptors.numel() == world_size * kDescriptorWords, "descriptor size mismatch");
    TORCH_CHECK(metadata.device() == descriptors.device(), "metadata tensors must share a device");
    TORCH_CHECK(max_numel > 0 && payload_stride >= 0, "invalid dynamic metadata bounds");
    c10::cuda::CUDAGuard guard(metadata.device());
    constexpr int threads = 128;
    const int blocks = static_cast<int>((world_size + threads - 1) / threads);
    decode_dynamic_metadata_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getCurrentCUDAStream(metadata.get_device())
    >>>(
        metadata.data_ptr<int64_t>(),
        descriptors.data_ptr<int64_t>(),
        world_size,
        dtype_code,
        bit,
        group_size,
        quant_type_code,
        compact ? 1 : 0,
        layout_generation,
        max_numel,
        payload_stride
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
