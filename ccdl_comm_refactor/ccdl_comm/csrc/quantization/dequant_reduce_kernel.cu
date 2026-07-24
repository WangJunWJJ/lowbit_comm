#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <vector>

#include "dequant_api.cuh"
#include "enum.cuh"
#include "utils.cuh"

namespace {

constexpr int kFusedGroupSize = 64;
constexpr int kFusedBit = 8;
constexpr int kFusedMaxInputs = 8;
constexpr int kThreadsPerBlock = 256;

template <typename scalar_t>
__device__ float read_scalar_as_float(const uint8_t* base, int64_t byte_offset) {
    return half2float<scalar_t>(*reinterpret_cast<const scalar_t*>(base + byte_offset));
}

__device__ uint8_t read_u8_packed16(const uint8_t* base, int64_t byte_offset, int64_t element_in_group) {
    uint16_t packed = *reinterpret_cast<const uint16_t*>(base + byte_offset + (element_in_group / 2) * sizeof(uint16_t));
    return (element_in_group & 1) ? static_cast<uint8_t>(packed & 0xFF) : static_cast<uint8_t>((packed >> 8) & 0xFF);
}

__device__ uint8_t read_u8_packed32(const uint8_t* base, int64_t byte_offset, int64_t element_in_group) {
    uint32_t packed = *reinterpret_cast<const uint32_t*>(base + byte_offset + (element_in_group / 4) * sizeof(uint32_t));
    int shift = (3 - static_cast<int>(element_in_group & 3)) * 8;
    return static_cast<uint8_t>((packed >> shift) & 0xFF);
}

template <typename scalar_t>
__device__ float dequant_one_16bit_scale(const uint8_t* input, int64_t group_id, int64_t element_in_group, bool compact, int64_t num_groups) {
    int64_t data_offset;
    int64_t scale_offset;
    if (compact) {
        constexpr int64_t bytes_per_group = kFusedGroupSize + sizeof(scalar_t);
        data_offset = group_id * bytes_per_group;
        scale_offset = data_offset + kFusedGroupSize;
    } else {
        data_offset = group_id * kFusedGroupSize;
        scale_offset = num_groups * kFusedGroupSize + group_id * sizeof(scalar_t);
    }
    uint8_t raw = read_u8_packed16(input, data_offset, element_in_group);
    float scale = read_scalar_as_float<scalar_t>(input, scale_offset) / 127.0f;
    return static_cast<float>(static_cast<int8_t>(raw)) * scale;
}

__device__ float dequant_one_fp32_scale(const uint8_t* input, int64_t group_id, int64_t element_in_group, bool compact, int64_t num_groups) {
    int64_t data_offset;
    int64_t scale_offset;
    if (compact) {
        constexpr int64_t bytes_per_group = kFusedGroupSize + sizeof(float);
        data_offset = group_id * bytes_per_group;
        scale_offset = data_offset + kFusedGroupSize;
    } else {
        data_offset = group_id * kFusedGroupSize;
        scale_offset = num_groups * kFusedGroupSize + group_id * sizeof(float);
    }
    uint8_t raw = read_u8_packed32(input, data_offset, element_in_group);
    float scale = *reinterpret_cast<const float*>(input + scale_offset) / 127.0f;
    return static_cast<float>(static_cast<int8_t>(raw)) * scale;
}

template <typename scalar_t>
__global__ void dequant_reduce_fused_16bit_kernel(
    const uint8_t* input0,
    const uint8_t* input1,
    const uint8_t* input2,
    const uint8_t* input3,
    const uint8_t* input4,
    const uint8_t* input5,
    const uint8_t* input6,
    const uint8_t* input7,
    int64_t num_inputs,
    scalar_t* output,
    int64_t numel,
    bool compact
) {
    const uint8_t* inputs[kFusedMaxInputs] = {input0, input1, input2, input3, input4, input5, input6, input7};
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t num_groups = (numel + kFusedGroupSize - 1) / kFusedGroupSize;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        int64_t group_id = index / kFusedGroupSize;
        int64_t element_in_group = index - group_id * kFusedGroupSize;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                sum += dequant_one_16bit_scale<scalar_t>(inputs[rank], group_id, element_in_group, compact, num_groups);
            }
        }
        output[index] = float2half<scalar_t>(sum);
    }
}

__global__ void dequant_reduce_fused_fp32_kernel(
    const uint8_t* input0,
    const uint8_t* input1,
    const uint8_t* input2,
    const uint8_t* input3,
    const uint8_t* input4,
    const uint8_t* input5,
    const uint8_t* input6,
    const uint8_t* input7,
    int64_t num_inputs,
    float* output,
    int64_t numel,
    bool compact
) {
    const uint8_t* inputs[kFusedMaxInputs] = {input0, input1, input2, input3, input4, input5, input6, input7};
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t num_groups = (numel + kFusedGroupSize - 1) / kFusedGroupSize;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        int64_t group_id = index / kFusedGroupSize;
        int64_t element_in_group = index - group_id * kFusedGroupSize;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                sum += dequant_one_fp32_scale(inputs[rank], group_id, element_in_group, compact, num_groups);
            }
        }
        output[index] = sum;
    }
}

std::array<const uint8_t*, kFusedMaxInputs> tensor_ptrs(const std::vector<torch::Tensor>& inputs) {
    std::array<const uint8_t*, kFusedMaxInputs> ptrs{};
    for (size_t i = 0; i < inputs.size(); ++i) {
        ptrs[i] = static_cast<const uint8_t*>(inputs[i].data_ptr());
    }
    return ptrs;
}

bool can_use_fused_dequant_reduce(
    const std::vector<torch::Tensor>& inputs,
    const torch::Tensor& output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type
) {
    if (inputs.empty() || inputs.size() > kFusedMaxInputs) return false;
    if (group_size != kFusedGroupSize || topk != 0 || bit != kFusedBit || quant_type != QuantType::Linear) return false;
    if (!output.is_cuda() || !output.is_contiguous()) return false;
    for (const auto& input : inputs) {
        if (!input.is_cuda() || !input.is_contiguous() || input.dtype() != torch::kUInt8) return false;
        if (input.device() != output.device()) return false;
    }
    return output.dtype() == torch::kHalf || output.dtype() == torch::kBFloat16 || output.dtype() == torch::kFloat32;
}

}  // namespace

bool try_inplace_dequantize_reduce_fused(
    std::vector<torch::Tensor> inputs,
    torch::Tensor output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact
) {
    if (!can_use_fused_dequant_reduce(inputs, output, group_size, topk, bit, quant_type)) {
        return false;
    }
    auto ptrs = tensor_ptrs(inputs);
    int64_t numel = output.numel();
    int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();
    if (output.dtype() == torch::kHalf) {
        dequant_reduce_fused_16bit_kernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()), static_cast<__half*>(output.data_ptr()), numel, compact
        );
        return true;
    }
    if (output.dtype() == torch::kBFloat16) {
        dequant_reduce_fused_16bit_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()), static_cast<__nv_bfloat16*>(output.data_ptr()), numel, compact
        );
        return true;
    }
    dequant_reduce_fused_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        static_cast<int64_t>(inputs.size()), static_cast<float*>(output.data_ptr()), numel, compact
    );
    return true;
}
