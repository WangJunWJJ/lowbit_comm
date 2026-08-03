#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
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
    bool compact,
    float inv_divisor
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
        output[index] = float2half<scalar_t>(sum * inv_divisor);
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
    bool compact,
    float inv_divisor
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
        output[index] = sum * inv_divisor;
    }
}

template <typename scalar_t>
__global__ void error_feedback_update_kernel(
    const scalar_t* prepared,
    const scalar_t* restored,
    scalar_t* residual,
    int64_t numel
) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        float value = half2float<scalar_t>(prepared[index]) - half2float<scalar_t>(restored[index]);
        residual[index] = float2half<scalar_t>(value);
    }
}

template <typename scalar_t>
__global__ void dequant_reduce_mean_feedback_fused_16bit_kernel(
    const uint8_t* input0,
    const uint8_t* input1,
    const uint8_t* input2,
    const uint8_t* input3,
    const uint8_t* input4,
    const uint8_t* input5,
    const uint8_t* input6,
    const uint8_t* input7,
    int64_t num_inputs,
    const scalar_t* prepared,
    scalar_t* restored,
    scalar_t* residual,
    int64_t numel,
    bool compact,
    float inv_divisor
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
        float restored_value = sum * inv_divisor;
        restored[index] = float2half<scalar_t>(restored_value);
        residual[index] = float2half<scalar_t>(half2float<scalar_t>(prepared[index]) - restored_value);
    }
}

__global__ void dequant_reduce_mean_feedback_fused_fp32_kernel(
    const uint8_t* input0,
    const uint8_t* input1,
    const uint8_t* input2,
    const uint8_t* input3,
    const uint8_t* input4,
    const uint8_t* input5,
    const uint8_t* input6,
    const uint8_t* input7,
    int64_t num_inputs,
    const float* prepared,
    float* restored,
    float* residual,
    int64_t numel,
    bool compact,
    float inv_divisor
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
        float restored_value = sum * inv_divisor;
        restored[index] = restored_value;
        residual[index] = prepared[index] - restored_value;
    }
}

__global__ void error_feedback_update_fp32_kernel(
    const float* prepared,
    const float* restored,
    float* residual,
    int64_t numel
) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        residual[index] = prepared[index] - restored[index];
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
    if (!output.is_cuda() || !output.is_contiguous() || output.numel() == 0 || output.numel() % kFusedGroupSize != 0) return false;
    int64_t num_groups = (output.numel() + kFusedGroupSize - 1) / kFusedGroupSize;
    int64_t expected_input_numel = num_groups * (kFusedGroupSize + output.element_size());
    for (const auto& input : inputs) {
        if (!input.is_cuda() || !input.is_contiguous() || input.dtype() != torch::kUInt8) return false;
        if (input.device() != output.device()) return false;
        if (input.numel() != expected_input_numel) return false;
    }
    return output.dtype() == torch::kHalf || output.dtype() == torch::kBFloat16 || output.dtype() == torch::kFloat32;
}

}  // namespace

void inplace_error_feedback_update(torch::Tensor prepared, torch::Tensor restored, torch::Tensor residual) {
    TORCH_CHECK(prepared.is_cuda(), "prepared must be a CUDA tensor");
    TORCH_CHECK(restored.is_cuda(), "restored must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(prepared.is_contiguous(), "prepared must be contiguous");
    TORCH_CHECK(restored.is_contiguous(), "restored must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(prepared.numel() == restored.numel(), "prepared and restored must have the same number of elements");
    TORCH_CHECK(prepared.numel() == residual.numel(), "prepared and residual must have the same number of elements");
    TORCH_CHECK(prepared.dtype() == restored.dtype(), "prepared and restored must have the same dtype");
    TORCH_CHECK(prepared.dtype() == residual.dtype(), "prepared and residual must have the same dtype");
    TORCH_CHECK(prepared.device() == restored.device(), "prepared and restored must be on the same device");
    TORCH_CHECK(prepared.device() == residual.device(), "prepared and residual must be on the same device");

    int64_t numel = prepared.numel();
    int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();

    if (prepared.dtype() == torch::kHalf) {
        error_feedback_update_kernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __half*>(prepared.data_ptr()),
            static_cast<const __half*>(restored.data_ptr()),
            static_cast<__half*>(residual.data_ptr()),
            numel
        );
        return;
    }
    if (prepared.dtype() == torch::kBFloat16) {
        error_feedback_update_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(prepared.data_ptr()),
            static_cast<const __nv_bfloat16*>(restored.data_ptr()),
            static_cast<__nv_bfloat16*>(residual.data_ptr()),
            numel
        );
        return;
    }
    if (prepared.dtype() == torch::kFloat32) {
        error_feedback_update_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const float*>(prepared.data_ptr()),
            static_cast<const float*>(restored.data_ptr()),
            static_cast<float*>(residual.data_ptr()),
            numel
        );
        return;
    }
    TORCH_CHECK(false, "unsupported dtype for inplace_error_feedback_update");
}

bool try_inplace_dequantize_reduce_fused(
    std::vector<torch::Tensor> inputs,
    torch::Tensor output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    float inv_divisor
) {
    if (!can_use_fused_dequant_reduce(inputs, output, group_size, topk, bit, quant_type)) {
        return false;
    }
    c10::cuda::CUDAGuard device_guard(output.device());
    auto ptrs = tensor_ptrs(inputs);
    int64_t numel = output.numel();
    int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();
    if (output.dtype() == torch::kHalf) {
        dequant_reduce_fused_16bit_kernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()), static_cast<__half*>(output.data_ptr()), numel, compact, inv_divisor
        );
        return true;
    }
    if (output.dtype() == torch::kBFloat16) {
        dequant_reduce_fused_16bit_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()), static_cast<__nv_bfloat16*>(output.data_ptr()), numel, compact, inv_divisor
        );
        return true;
    }
    dequant_reduce_fused_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        static_cast<int64_t>(inputs.size()), static_cast<float*>(output.data_ptr()), numel, compact, inv_divisor
    );
    return true;
}

bool inplace_dequantize_reduce_mean_update_error_feedback(
    std::vector<torch::Tensor> inputs,
    torch::Tensor prepared,
    torch::Tensor restored,
    torch::Tensor residual,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    int64_t divisor
) {
    TORCH_CHECK(divisor > 0, "divisor must be > 0");
    TORCH_CHECK(prepared.is_cuda(), "prepared must be a CUDA tensor");
    TORCH_CHECK(restored.is_cuda(), "restored must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(prepared.is_contiguous(), "prepared must be contiguous");
    TORCH_CHECK(restored.is_contiguous(), "restored must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(prepared.numel() == residual.numel(), "prepared and residual must have the same number of elements");
    TORCH_CHECK(restored.numel() >= prepared.numel(), "restored must have at least prepared.numel() elements");
    TORCH_CHECK(prepared.dtype() == restored.dtype(), "prepared and restored must have the same dtype");
    TORCH_CHECK(prepared.dtype() == residual.dtype(), "prepared and residual must have the same dtype");
    TORCH_CHECK(prepared.device() == restored.device(), "prepared and restored must be on the same device");
    TORCH_CHECK(prepared.device() == residual.device(), "prepared and residual must be on the same device");

    if (!can_use_fused_dequant_reduce(inputs, restored, group_size, topk, bit, quant_type)) {
        return false;
    }

    c10::cuda::CUDAGuard device_guard(restored.device());
    auto ptrs = tensor_ptrs(inputs);
    int64_t numel = prepared.numel();
    int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();
    float inv_divisor = 1.0f / static_cast<float>(divisor);

    if (prepared.dtype() == torch::kHalf) {
        dequant_reduce_mean_feedback_fused_16bit_kernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()),
            static_cast<const __half*>(prepared.data_ptr()),
            static_cast<__half*>(restored.data_ptr()),
            static_cast<__half*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
        return true;
    }
    if (prepared.dtype() == torch::kBFloat16) {
        dequant_reduce_mean_feedback_fused_16bit_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()),
            static_cast<const __nv_bfloat16*>(prepared.data_ptr()),
            static_cast<__nv_bfloat16*>(restored.data_ptr()),
            static_cast<__nv_bfloat16*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
        return true;
    }
    dequant_reduce_mean_feedback_fused_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        static_cast<int64_t>(inputs.size()),
        static_cast<const float*>(prepared.data_ptr()),
        static_cast<float*>(restored.data_ptr()),
        static_cast<float*>(residual.data_ptr()),
        numel,
        compact,
        inv_divisor
    );
    return true;
}
