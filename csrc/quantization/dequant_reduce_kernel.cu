#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include "dequant_api.cuh"
#include "enum.cuh"
#include "utils.cuh"

namespace {

constexpr int kFusedGroupSize = 64;
constexpr int kFusedBit = 8;
constexpr int kFusedMaxInputs = 8;
constexpr int kThreadsPerBlock = 256;

struct ShardDequantGate {
    std::atomic<bool> armed{false};
    std::mutex mutex;
    bool enqueued{false};
};

ShardDequantGate& shard_dequant_gate() {
    static ShardDequantGate* gate = new ShardDequantGate();
    return *gate;
}

__global__ void delay_shard_dequant_completion(
    unsigned long long delay_cycles
) {
    const unsigned long long start = clock64();
    while (clock64() - start < delay_cycles) {
        __nanosleep(1000);
    }
}

void maybe_enqueue_shard_dequant_gate(cudaStream_t current_stream) {
    ShardDequantGate& gate = shard_dequant_gate();
    if (!gate.armed.load(std::memory_order_acquire)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(gate.mutex);
        TORCH_CHECK(!gate.enqueued, "shard dequant test gate is already used");
        gate.enqueued = true;
    }
    int device = 0;
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    constexpr unsigned long long kDelayMilliseconds = 500;
    const auto delay_cycles =
        static_cast<unsigned long long>(properties.clockRate) *
        kDelayMilliseconds;
    delay_shard_dequant_completion<<<1, 1, 0, current_stream>>>(delay_cycles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t checked_shard_mul(int64_t left, int64_t right) {
    TORCH_CHECK(
        left >= 0 && right >= 0,
        "shard dequant-reduce size is invalid"
    );
    TORCH_CHECK(
        left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
        "shard dequant-reduce size overflow"
    );
    return left * right;
}

template <typename scalar_t>
__device__ float read_scalar_as_float(const uint8_t* base, int64_t byte_offset) {
    return half2float<scalar_t>(*reinterpret_cast<const scalar_t*>(base + byte_offset));
}

template <typename scalar_t>
__device__ float dequant_one_shard_payload(
    const uint8_t* input,
    int64_t source_rank,
    int64_t payload_bytes,
    int64_t group_id,
    int64_t element_in_group,
    int64_t group_size
) {
    const int64_t bytes_per_group =
        group_size + static_cast<int64_t>(sizeof(scalar_t));
    const int64_t group_offset =
        source_rank * payload_bytes + group_id * bytes_per_group;
    const float scale =
        read_scalar_as_float<scalar_t>(input, group_offset) / 127.0f;
    const int8_t quantized = *reinterpret_cast<const int8_t*>(
        input + group_offset + sizeof(scalar_t) + element_in_group
    );
    return static_cast<float>(quantized) * scale;
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

template <typename scalar_t>
__device__ float dequant_one_16bit_scale_runtime(
    const uint8_t* input,
    int64_t group_id,
    int64_t element_in_group,
    bool compact,
    int64_t num_groups,
    int64_t group_size
) {
    int64_t data_offset;
    int64_t scale_offset;
    if (compact) {
        const int64_t bytes_per_group = group_size + sizeof(scalar_t);
        data_offset = group_id * bytes_per_group;
        scale_offset = data_offset + group_size;
    } else {
        data_offset = group_id * group_size;
        scale_offset = num_groups * group_size + group_id * sizeof(scalar_t);
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

__device__ float dequant_one_fp32_scale_runtime(
    const uint8_t* input,
    int64_t group_id,
    int64_t element_in_group,
    bool compact,
    int64_t num_groups,
    int64_t group_size
) {
    int64_t data_offset;
    int64_t scale_offset;
    if (compact) {
        const int64_t bytes_per_group = group_size + sizeof(float);
        data_offset = group_id * bytes_per_group;
        scale_offset = data_offset + group_size;
    } else {
        data_offset = group_id * group_size;
        scale_offset = num_groups * group_size + group_id * sizeof(float);
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
    int64_t group_size,
    bool compact,
    float inv_divisor
) {
    const uint8_t* inputs[kFusedMaxInputs] = {input0, input1, input2, input3, input4, input5, input6, input7};
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t num_groups = (numel + group_size - 1) / group_size;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        int64_t group_id = index / group_size;
        int64_t element_in_group = index - group_id * group_size;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                sum += dequant_one_16bit_scale_runtime<scalar_t>(
                    inputs[rank], group_id, element_in_group, compact,
                    num_groups, group_size);
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
    int64_t group_size,
    bool compact,
    float inv_divisor
) {
    const uint8_t* inputs[kFusedMaxInputs] = {input0, input1, input2, input3, input4, input5, input6, input7};
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t num_groups = (numel + group_size - 1) / group_size;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        int64_t group_id = index / group_size;
        int64_t element_in_group = index - group_id * group_size;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                sum += dequant_one_fp32_scale_runtime(
                    inputs[rank], group_id, element_in_group, compact,
                    num_groups, group_size);
            }
        }
        output[index] = sum * inv_divisor;
    }
}

template <typename scalar_t>
__global__ void shard_dequant_reduce_kernel(
    const uint8_t* input,
    scalar_t* output,
    int64_t logical_shard_length,
    int64_t valid_length,
    int64_t world_size,
    int64_t group_size,
    int64_t payload_bytes,
    float inv_divisor
) {
    int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (
        ; index < logical_shard_length;
        index += static_cast<int64_t>(blockDim.x) * gridDim.x
    ) {
        if (index >= valid_length) {
            output[index] = float2half<scalar_t>(0.0f);
            continue;
        }
        const int64_t group_id = index / group_size;
        const int64_t element_in_group = index - group_id * group_size;
        float sum = 0.0f;
        for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
            sum += dequant_one_shard_payload<scalar_t>(
                input,
                source_rank,
                payload_bytes,
                group_id,
                element_in_group,
                group_size
            );
        }
        output[index] = float2half<scalar_t>(sum * inv_divisor);
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
    int64_t local_input_index,
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
        float local_restored = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                float value = dequant_one_16bit_scale<scalar_t>(inputs[rank], group_id, element_in_group, compact, num_groups);
                sum += value;
                if (rank == local_input_index) local_restored = value;
            }
        }
        float restored_value = sum * inv_divisor;
        restored[index] = float2half<scalar_t>(restored_value);
        residual[index] = float2half<scalar_t>(half2float<scalar_t>(prepared[index]) - local_restored);
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
    int64_t local_input_index,
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
        float local_restored = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                float value = dequant_one_fp32_scale(inputs[rank], group_id, element_in_group, compact, num_groups);
                sum += value;
                if (rank == local_input_index) local_restored = value;
            }
        }
        float restored_value = sum * inv_divisor;
        restored[index] = restored_value;
        residual[index] = prepared[index] - local_restored;
    }
}

template <typename scalar_t>
__global__ void dequant_reduce_mean_feedback_global_16bit_kernel(
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
    const uint8_t* inputs[kFusedMaxInputs] = {
        input0, input1, input2, input3, input4, input5, input6, input7
    };
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t num_groups =
        (numel + kFusedGroupSize - 1) / kFusedGroupSize;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        const int64_t group_id = index / kFusedGroupSize;
        const int64_t element_in_group = index - group_id * kFusedGroupSize;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                sum += dequant_one_16bit_scale<scalar_t>(
                    inputs[rank],
                    group_id,
                    element_in_group,
                    compact,
                    num_groups
                );
            }
        }
        const float restored_value = sum * inv_divisor;
        restored[index] = float2half<scalar_t>(restored_value);
        residual[index] = float2half<scalar_t>(
            half2float<scalar_t>(prepared[index]) - restored_value
        );
    }
}

__global__ void dequant_reduce_mean_feedback_global_fp32_kernel(
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
    const uint8_t* inputs[kFusedMaxInputs] = {
        input0, input1, input2, input3, input4, input5, input6, input7
    };
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t num_groups =
        (numel + kFusedGroupSize - 1) / kFusedGroupSize;
    for (; index < numel; index += blockDim.x * gridDim.x) {
        const int64_t group_id = index / kFusedGroupSize;
        const int64_t element_in_group = index - group_id * kFusedGroupSize;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t rank = 0; rank < kFusedMaxInputs; ++rank) {
            if (rank < num_inputs) {
                sum += dequant_one_fp32_scale(
                    inputs[rank],
                    group_id,
                    element_in_group,
                    compact,
                    num_groups
                );
            }
        }
        const float restored_value = sum * inv_divisor;
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
    if ((group_size != 16 && group_size != 32 && group_size != 64) || topk != 0 || bit != kFusedBit || quant_type != QuantType::Linear) return false;
    if (!output.is_cuda() || !output.is_contiguous() || output.numel() == 0) return false;
    int64_t num_groups = (output.numel() + group_size - 1) / group_size;
    int64_t expected_input_numel = num_groups * (group_size + output.element_size());
    for (const auto& input : inputs) {
        if (!input.is_cuda() || !input.is_contiguous() || input.dtype() != torch::kUInt8) return false;
        if (input.device() != output.device()) return false;
        if (input.numel() != expected_input_numel) return false;
    }
    return output.dtype() == torch::kHalf || output.dtype() == torch::kBFloat16 || output.dtype() == torch::kFloat32;
}

bool can_use_shard_dequant_reduce(
    const torch::Tensor& input,
    const torch::Tensor& output,
    int64_t logical_shard_length,
    int64_t valid_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size,
    float inv_divisor,
    int64_t& payload_bytes
) {
    if (
        output.dtype() != torch::kHalf &&
        output.dtype() != torch::kBFloat16
    ) {
        return false;
    }
    if (
        (group_size != 16 && group_size != 32 && group_size != 64) ||
        world_size <= 0 || logical_shard_length <= 0 ||
        valid_length < 0 || valid_length > logical_shard_length ||
        !std::isfinite(inv_divisor) || inv_divisor <= 0.0f
    ) {
        return false;
    }
    TORCH_CHECK(input.is_cuda(), "shard payload must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "shard output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "shard payload must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "shard output must be contiguous");
    TORCH_CHECK(
        input.dtype() == torch::kUInt8,
        "shard payload must have uint8 dtype"
    );
    TORCH_CHECK(
        input.device() == output.device(),
        "shard payload and output must be on the same device"
    );
    TORCH_CHECK(input.dim() == 1, "shard payload must be one-dimensional");
    TORCH_CHECK(output.dim() == 1, "shard output must be one-dimensional");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint16_t) == 0,
        "shard payload data must be aligned to 2 bytes"
    );
    TORCH_CHECK(
        output.numel() == logical_shard_length,
        "shard output length does not match the logical shard"
    );
    const int64_t groups_per_shard =
        logical_shard_length / group_size +
        (logical_shard_length % group_size != 0);
    const int64_t expected_transport =
        checked_shard_mul(groups_per_shard, group_size);
    TORCH_CHECK(
        transport_shard_length == expected_transport,
        "shard transport length does not match the logical shard"
    );
    payload_bytes = checked_shard_mul(
        groups_per_shard,
        group_size + static_cast<int64_t>(sizeof(uint16_t))
    );
    const int64_t expected_input =
        checked_shard_mul(world_size, payload_bytes);
    TORCH_CHECK(
        input.numel() == expected_input,
        "shard payload size does not match the transport layout"
    );
    return true;
}

template <typename scalar_t>
__device__ int quantize_linear_int8(float value, float multiplier) {
    return clamp_and_round<float>(value * multiplier, -127, 127);
}

template <typename scalar_t>
__global__ void dequant_reduce_mean_requantize_kernel(
    const uint8_t* input0,
    const uint8_t* input1,
    const uint8_t* input2,
    const uint8_t* input3,
    const uint8_t* input4,
    const uint8_t* input5,
    const uint8_t* input6,
    const uint8_t* input7,
    int64_t num_inputs,
    uint8_t* output,
    int64_t num_groups,
    int64_t expected_output_numel,
    int64_t output_numel,
    float inv_divisor
) {
    __shared__ scalar_t reduced[kFusedGroupSize];
    __shared__ float maxima[kFusedGroupSize];
    const uint8_t* inputs[kFusedMaxInputs] = {
        input0, input1, input2, input3, input4, input5, input6, input7
    };
    const int lane = threadIdx.x;
    const int64_t group = blockIdx.x;

    float sum = 0.0f;
    #pragma unroll
    for (int rank = 0; rank < kFusedMaxInputs; ++rank) {
        if (rank < num_inputs) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                sum += dequant_one_fp32_scale(inputs[rank], group, lane, false, num_groups);
            } else {
                sum += dequant_one_16bit_scale<scalar_t>(inputs[rank], group, lane, false, num_groups);
            }
        }
    }
    const scalar_t rounded = float2half<scalar_t>(sum * inv_divisor);
    const float rounded_value = half2float<scalar_t>(rounded);
    reduced[lane] = rounded;
    maxima[lane] = isfinite(rounded_value)
        ? fabsf(rounded_value)
        : non_finite_quant_scale();
    __syncthreads();

    for (int offset = kFusedGroupSize / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        }
        __syncthreads();
    }

    const scalar_t stored_scale = float2half<scalar_t>(fmaxf(maxima[0], 1.0e-6f));
    const float scale = half2float<scalar_t>(stored_scale);
    const float multiplier = 127.0f / scale;
    uint8_t* group_output = output + group * kFusedGroupSize;
    if constexpr (sizeof(scalar_t) == sizeof(uint16_t)) {
        if ((lane & 1) == 0) {
            const uint16_t first = static_cast<uint16_t>(
                static_cast<uint32_t>(quantize_linear_int8<scalar_t>(half2float<scalar_t>(reduced[lane]), multiplier)) & 0xff
            );
            const uint16_t second = static_cast<uint16_t>(
                static_cast<uint32_t>(quantize_linear_int8<scalar_t>(half2float<scalar_t>(reduced[lane + 1]), multiplier)) & 0xff
            );
            reinterpret_cast<uint16_t*>(group_output)[lane / 2] = static_cast<uint16_t>((first << 8) | second);
        }
    } else if ((lane & 3) == 0) {
        uint32_t packed = 0;
        #pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
            const uint32_t quantized = static_cast<uint32_t>(
                quantize_linear_int8<scalar_t>(half2float<scalar_t>(reduced[lane + offset]), multiplier)
            ) & 0xff;
            packed = (packed << 8) | quantized;
        }
        reinterpret_cast<uint32_t*>(group_output)[lane / 4] = packed;
    }
    if (lane == 0) {
        *reinterpret_cast<scalar_t*>(
            output + num_groups * kFusedGroupSize + group * sizeof(scalar_t)
        ) = stored_scale;
    }
    if (group == 0) {
        for (int64_t index = expected_output_numel + lane; index < output_numel; index += kFusedGroupSize) {
            output[index] = 0;
        }
    }
}

bool can_use_fused_requantize(
    const std::vector<torch::Tensor>& inputs,
    const torch::Tensor& output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    DType dtype,
    int64_t& num_groups,
    int64_t& expected_output_numel
) {
    if (inputs.empty() || inputs.size() > kFusedMaxInputs) return false;
    if (group_size != kFusedGroupSize || topk != 0 || bit != kFusedBit) return false;
    if (quant_type != QuantType::Linear || compact) return false;
    if (!output.is_cuda() || !output.is_contiguous() || output.dtype() != torch::kUInt8) return false;
    const int64_t scale_bytes = dtype == DType::FP32 ? sizeof(float) : sizeof(uint16_t);
    const int64_t bytes_per_group = kFusedGroupSize + scale_bytes;
    if (inputs[0].numel() == 0 || inputs[0].numel() % bytes_per_group != 0) return false;
    num_groups = inputs[0].numel() / bytes_per_group;
    expected_output_numel = num_groups * bytes_per_group;
    if (output.numel() < expected_output_numel || output.numel() % 16 != 0) return false;
    for (const auto& input : inputs) {
        if (!input.is_cuda() || !input.is_contiguous() || input.dtype() != torch::kUInt8) return false;
        if (input.device() != output.device() || input.numel() != expected_output_numel) return false;
    }
    return true;
}

template <typename scalar_t>
__global__ void dequantize_gathered_kernel(
    const uint8_t* input,
    scalar_t* output,
    int64_t world_size,
    int64_t payload_stride,
    int64_t shard_numel,
    bool compact
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_numel = world_size * shard_numel;
    const int64_t num_groups = (shard_numel + kFusedGroupSize - 1) / kFusedGroupSize;
    for (; index < output_numel; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t rank = index / shard_numel;
        const int64_t local_index = index - rank * shard_numel;
        const int64_t group = local_index / kFusedGroupSize;
        const int64_t element = local_index - group * kFusedGroupSize;
        const uint8_t* payload = input + rank * payload_stride;
        float value;
        if constexpr (std::is_same<scalar_t, float>::value) {
            value = dequant_one_fp32_scale(payload, group, element, compact, num_groups);
        } else {
            value = dequant_one_16bit_scale<scalar_t>(payload, group, element, compact, num_groups);
        }
        output[index] = float2half<scalar_t>(value);
    }
}

template <typename scalar_t>
__global__ void dequantize_gathered_add_kernel(
    const uint8_t* input,
    scalar_t* output,
    int64_t payload_stride,
    int64_t shard_numel,
    int64_t original_numel
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num_groups =
        (shard_numel + kFusedGroupSize - 1) / kFusedGroupSize;
    for (
        ; index < original_numel;
        index += static_cast<int64_t>(blockDim.x) * gridDim.x
    ) {
        const int64_t rank = index / shard_numel;
        const int64_t local_index = index - rank * shard_numel;
        const int64_t group = local_index / kFusedGroupSize;
        const int64_t element = local_index - group * kFusedGroupSize;
        const uint8_t* payload = input + rank * payload_stride;
        const float delta = dequant_one_fp32_scale(
            payload,
            group,
            element,
            true,
            num_groups
        );
        const float updated = __fadd_rn(half2float(output[index]), delta);
        output[index] = float2half<scalar_t>(updated);
    }
}

bool can_use_fused_gathered_dequantize(
    const torch::Tensor& input,
    const torch::Tensor& output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    DType dtype,
    int64_t world_size,
    int64_t payload_numel,
    int64_t payload_stride,
    int64_t shard_numel
) {
    if (world_size < 1 || world_size > kFusedMaxInputs) return false;
    if (group_size != kFusedGroupSize || topk != 0 || bit != kFusedBit) return false;
    if (quant_type != QuantType::Linear) return false;
    if (shard_numel <= 0) return false;
    if (payload_stride < payload_numel || payload_stride % 16 != 0) return false;
    const int64_t scale_bytes = dtype == DType::FP32 ? sizeof(float) : sizeof(uint16_t);
    const int64_t num_groups = (shard_numel + kFusedGroupSize - 1) / kFusedGroupSize;
    if (payload_numel != num_groups * (kFusedGroupSize + scale_bytes)) return false;
    if (!input.is_cuda() || !input.is_contiguous() || input.dtype() != torch::kUInt8) return false;
    if (!output.is_cuda() || !output.is_contiguous() || input.device() != output.device()) return false;
    if (input.numel() != world_size * payload_stride) return false;
    if (output.numel() != world_size * shard_numel) return false;
    if (dtype == DType::FP16) return output.dtype() == torch::kHalf;
    if (dtype == DType::BF16) return output.dtype() == torch::kBFloat16;
    return output.dtype() == torch::kFloat32;
}

}  // namespace

bool inplace_dequantize_reduce_mean_requantize(
    std::vector<torch::Tensor> inputs,
    torch::Tensor output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    DType dtype,
    int64_t divisor
) {
    TORCH_CHECK(divisor > 0, "divisor must be > 0");
    int64_t num_groups = 0;
    int64_t expected_output_numel = 0;
    if (!can_use_fused_requantize(
        inputs,
        output,
        group_size,
        topk,
        bit,
        quant_type,
        compact,
        dtype,
        num_groups,
        expected_output_numel
    )) {
        return false;
    }
    c10::cuda::CUDAGuard device_guard(output.device());
    auto ptrs = tensor_ptrs(inputs);
    cudaStream_t stream = get_current_cuda_stream();
    float inv_divisor = 1.0f / static_cast<float>(divisor);
    if (dtype == DType::FP16) {
        dequant_reduce_mean_requantize_kernel<__half><<<num_groups, kFusedGroupSize, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            inputs.size(), static_cast<uint8_t*>(output.data_ptr()), num_groups,
            expected_output_numel, output.numel(), inv_divisor
        );
    } else if (dtype == DType::BF16) {
        dequant_reduce_mean_requantize_kernel<__nv_bfloat16><<<num_groups, kFusedGroupSize, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            inputs.size(), static_cast<uint8_t*>(output.data_ptr()), num_groups,
            expected_output_numel, output.numel(), inv_divisor
        );
    } else {
        dequant_reduce_mean_requantize_kernel<float><<<num_groups, kFusedGroupSize, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            inputs.size(), static_cast<uint8_t*>(output.data_ptr()), num_groups,
            expected_output_numel, output.numel(), inv_divisor
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool inplace_dequantize_gathered(
    torch::Tensor input,
    torch::Tensor output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    DType dtype,
    int64_t world_size,
    int64_t payload_numel,
    int64_t payload_stride,
    int64_t shard_numel
) {
    if (!can_use_fused_gathered_dequantize(
        input,
        output,
        group_size,
        topk,
        bit,
        quant_type,
        compact,
        dtype,
        world_size,
        payload_numel,
        payload_stride,
        shard_numel
    )) {
        return false;
    }
    c10::cuda::CUDAGuard device_guard(output.device());
    const int64_t output_numel = output.numel();
    int64_t blocks = (output_numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();
    if (dtype == DType::FP16) {
        dequantize_gathered_kernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<__half*>(output.data_ptr()),
            world_size,
            payload_stride,
            shard_numel,
            compact
        );
    } else if (dtype == DType::BF16) {
        dequantize_gathered_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            world_size,
            payload_stride,
            shard_numel,
            compact
        );
    } else {
        dequantize_gathered_kernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            world_size,
            payload_stride,
            shard_numel,
            compact
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool inplace_dequantize_gathered_add(
    torch::Tensor input,
    torch::Tensor output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    int64_t world_size,
    int64_t payload_numel,
    int64_t payload_stride,
    int64_t shard_numel,
    int64_t original_numel
) {
    if (
        group_size != kFusedGroupSize || topk != 0 || bit != kFusedBit ||
        quant_type != QuantType::Linear || !compact || world_size <= 0 ||
        shard_numel <= 0 || original_numel < 0 ||
        original_numel > world_size * shard_numel ||
        payload_stride < payload_numel || payload_stride % alignof(float) != 0
    ) {
        return false;
    }
    const int64_t num_groups =
        (shard_numel + kFusedGroupSize - 1) / kFusedGroupSize;
    if (payload_numel != num_groups * (kFusedGroupSize + static_cast<int64_t>(sizeof(float)))) {
        return false;
    }
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must have uint8 dtype");
    TORCH_CHECK(input.device() == output.device(), "input and output must be on the same device");
    TORCH_CHECK(
        input.numel() == world_size * payload_stride,
        "input has an invalid gathered payload size"
    );
    TORCH_CHECK(
        output.numel() == world_size * shard_numel,
        "output has an invalid gathered shard size"
    );
    if (
        output.dtype() != torch::kHalf &&
        output.dtype() != torch::kBFloat16 &&
        output.dtype() != torch::kFloat32
    ) {
        return false;
    }
    if (original_numel == 0) {
        return true;
    }

    c10::cuda::CUDAGuard device_guard(output.device());
    int64_t blocks = (original_numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();
    if (output.dtype() == torch::kHalf) {
        dequantize_gathered_add_kernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<__half*>(output.data_ptr()),
            payload_stride,
            shard_numel,
            original_numel
        );
    } else if (output.dtype() == torch::kBFloat16) {
        dequantize_gathered_add_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            payload_stride,
            shard_numel,
            original_numel
        );
    } else {
        dequantize_gathered_add_kernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            payload_stride,
            shard_numel,
            original_numel
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

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
    c10::cuda::CUDAGuard device_guard(prepared.device());

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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    if (prepared.dtype() == torch::kBFloat16) {
        error_feedback_update_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(prepared.data_ptr()),
            static_cast<const __nv_bfloat16*>(restored.data_ptr()),
            static_cast<__nv_bfloat16*>(residual.data_ptr()),
            numel
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    if (prepared.dtype() == torch::kFloat32) {
        error_feedback_update_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const float*>(prepared.data_ptr()),
            static_cast<const float*>(restored.data_ptr()),
            static_cast<float*>(residual.data_ptr()),
            numel
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    TORCH_CHECK(false, "unsupported dtype for inplace_error_feedback_update");
}

bool try_inplace_shard_dequantize_reduce(
    const torch::Tensor& input,
    torch::Tensor& output,
    int64_t logical_shard_length,
    int64_t valid_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size,
    float inv_divisor
) {
    int64_t payload_bytes = 0;
    if (!can_use_shard_dequant_reduce(
            input,
            output,
            logical_shard_length,
            valid_length,
            transport_shard_length,
            world_size,
            group_size,
            inv_divisor,
            payload_bytes)) {
        return false;
    }
    c10::cuda::CUDAGuard device_guard(output.device());
    const int64_t blocks = std::min<int64_t>(
        logical_shard_length / kThreadsPerBlock +
            (logical_shard_length % kThreadsPerBlock != 0),
        65535
    );
    cudaStream_t stream = get_current_cuda_stream();
    if (output.dtype() == torch::kHalf) {
        shard_dequant_reduce_kernel<__half><<<
            blocks, kThreadsPerBlock, 0, stream
        >>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<__half*>(output.data_ptr()),
            logical_shard_length,
            valid_length,
            world_size,
            group_size,
            payload_bytes,
            inv_divisor
        );
    } else {
        shard_dequant_reduce_kernel<__nv_bfloat16><<<
            blocks, kThreadsPerBlock, 0, stream
        >>>(
            static_cast<const uint8_t*>(input.data_ptr()),
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            logical_shard_length,
            valid_length,
            world_size,
            group_size,
            payload_bytes,
            inv_divisor
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    maybe_enqueue_shard_dequant_gate(stream);
    return true;
}

void arm_shard_dequant_gate_for_test() {
    ShardDequantGate& gate = shard_dequant_gate();
    std::lock_guard<std::mutex> lock(gate.mutex);
    TORCH_CHECK(
        !gate.armed.load(std::memory_order_relaxed),
        "shard dequant test gate is already armed"
    );
    gate.enqueued = false;
    gate.armed.store(true, std::memory_order_release);
}

void release_shard_dequant_gate_for_test() {
    ShardDequantGate& gate = shard_dequant_gate();
    std::lock_guard<std::mutex> lock(gate.mutex);
    TORCH_CHECK(
        gate.armed.load(std::memory_order_relaxed),
        "shard dequant test gate is not armed"
    );
    gate.armed.store(false, std::memory_order_release);
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
            static_cast<int64_t>(inputs.size()), static_cast<__half*>(output.data_ptr()), numel, group_size, compact, inv_divisor
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return true;
    }
    if (output.dtype() == torch::kBFloat16) {
        dequant_reduce_fused_16bit_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()), static_cast<__nv_bfloat16*>(output.data_ptr()), numel, group_size, compact, inv_divisor
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return true;
    }
    dequant_reduce_fused_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        static_cast<int64_t>(inputs.size()), static_cast<float*>(output.data_ptr()), numel, group_size, compact, inv_divisor
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool inplace_dequantize_reduce_update_local_error_feedback(
    std::vector<torch::Tensor> inputs,
    int64_t local_input_index,
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
    TORCH_CHECK(local_input_index >= 0 && local_input_index < inputs.size(), "local_input_index is out of range");
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
            local_input_index,
            static_cast<const __half*>(prepared.data_ptr()),
            static_cast<__half*>(restored.data_ptr()),
            static_cast<__half*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return true;
    }
    if (prepared.dtype() == torch::kBFloat16) {
        dequant_reduce_mean_feedback_fused_16bit_kernel<__nv_bfloat16><<<blocks, kThreadsPerBlock, 0, stream>>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()),
            local_input_index,
            static_cast<const __nv_bfloat16*>(prepared.data_ptr()),
            static_cast<__nv_bfloat16*>(restored.data_ptr()),
            static_cast<__nv_bfloat16*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return true;
    }
    dequant_reduce_mean_feedback_fused_fp32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        static_cast<int64_t>(inputs.size()),
        local_input_index,
        static_cast<const float*>(prepared.data_ptr()),
        static_cast<float*>(restored.data_ptr()),
        static_cast<float*>(residual.data_ptr()),
        numel,
        compact,
        inv_divisor
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    if (!can_use_fused_dequant_reduce(
        inputs,
        restored,
        group_size,
        topk,
        bit,
        quant_type
    )) {
        return false;
    }

    c10::cuda::CUDAGuard device_guard(restored.device());
    auto ptrs = tensor_ptrs(inputs);
    const int64_t numel = prepared.numel();
    int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::min<int64_t>(blocks, 65535);
    cudaStream_t stream = get_current_cuda_stream();
    const float inv_divisor = 1.0f / static_cast<float>(divisor);
    if (prepared.dtype() == torch::kHalf) {
        dequant_reduce_mean_feedback_global_16bit_kernel<__half><<<
            blocks, kThreadsPerBlock, 0, stream
        >>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3],
            ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()),
            static_cast<const __half*>(prepared.data_ptr()),
            static_cast<__half*>(restored.data_ptr()),
            static_cast<__half*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
    } else if (prepared.dtype() == torch::kBFloat16) {
        dequant_reduce_mean_feedback_global_16bit_kernel<__nv_bfloat16><<<
            blocks, kThreadsPerBlock, 0, stream
        >>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3],
            ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()),
            static_cast<const __nv_bfloat16*>(prepared.data_ptr()),
            static_cast<__nv_bfloat16*>(restored.data_ptr()),
            static_cast<__nv_bfloat16*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
    } else {
        dequant_reduce_mean_feedback_global_fp32_kernel<<<
            blocks, kThreadsPerBlock, 0, stream
        >>>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3],
            ptrs[4], ptrs[5], ptrs[6], ptrs[7],
            static_cast<int64_t>(inputs.size()),
            static_cast<const float*>(prepared.data_ptr()),
            static_cast<float*>(restored.data_ptr()),
            static_cast<float*>(residual.data_ptr()),
            numel,
            compact,
            inv_divisor
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}
