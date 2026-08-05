#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

#include "quant_api.cuh"
#include "utils.cuh"

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        return value;
    }
    return half2float<scalar_t>(value);
}

template <typename scalar_t, int GroupSize, int Bit>
__global__ void quantize_pack_kernel(
    const scalar_t* input,
    const scalar_t* residual,
    uint8_t* output,
    int64_t numel
) {
    constexpr int values_per_lane = 16;
    constexpr int lanes_per_group = GroupSize / values_per_lane;
    constexpr int groups_per_block = kThreads / lanes_per_group;
    static_assert(GroupSize % values_per_lane == 0, "group size must be divisible by values per lane");
    static_assert(lanes_per_group <= 32, "one group must fit in one warp");

    const int lane = threadIdx.x & (lanes_per_group - 1);
    const int group_in_block = threadIdx.x / lanes_per_group;
    const int64_t group = static_cast<int64_t>(blockIdx.x) * groups_per_block + group_in_block;
    const int64_t num_groups = (numel + GroupSize - 1) / GroupSize;
    const bool group_is_valid = group < num_groups;
    const int64_t lane_start = group * GroupSize + lane * values_per_lane;

    scalar_t prepared[values_per_lane];
    float max_abs = 0.0f;
    constexpr int values_per_vector = sizeof(int4) / sizeof(scalar_t);
    #pragma unroll
    for (int index = 0; index < values_per_lane; index += values_per_vector) {
        const int64_t global_index = lane_start + index;
        alignas(16) scalar_t input_values[values_per_vector];
        alignas(16) scalar_t residual_values[values_per_vector];
        const scalar_t* vector_input = group_is_valid ? input + global_index : input;
        const scalar_t* vector_residual = residual == nullptr
            ? nullptr
            : (group_is_valid ? residual + global_index : residual);
        const bool input_is_aligned =
            (reinterpret_cast<uintptr_t>(vector_input) & (alignof(int4) - 1)) == 0;
        const bool residual_is_aligned = residual == nullptr ||
            (reinterpret_cast<uintptr_t>(vector_residual) & (alignof(int4) - 1)) == 0;
        if (
            group_is_valid &&
            global_index + values_per_vector <= numel &&
            input_is_aligned &&
            residual_is_aligned
        ) {
            *reinterpret_cast<int4*>(input_values) = *reinterpret_cast<const int4*>(vector_input);
            if (residual != nullptr) {
                *reinterpret_cast<int4*>(residual_values) = *reinterpret_cast<const int4*>(vector_residual);
            }
        } else {
            #pragma unroll
            for (int offset = 0; offset < values_per_vector; ++offset) {
                const int64_t tail_index = global_index + offset;
                input_values[offset] = group_is_valid && tail_index < numel
                    ? input[tail_index]
                    : float2half<scalar_t>(0.0f);
                if (residual != nullptr) {
                    residual_values[offset] = group_is_valid && tail_index < numel
                        ? residual[tail_index]
                        : float2half<scalar_t>(0.0f);
                }
            }
        }
        #pragma unroll
        for (int offset = 0; offset < values_per_vector; ++offset) {
            float value = to_float(input_values[offset]);
            if (residual != nullptr) {
                value += to_float(residual_values[offset]);
            }
            prepared[index + offset] = float2half<scalar_t>(value);
            max_abs = fmaxf(max_abs, fabsf(to_float(prepared[index + offset])));
        }
    }

    #pragma unroll
    for (int offset = lanes_per_group / 2; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, offset, lanes_per_group));
    }

    const scalar_t stored_scale = float2half<scalar_t>(max_abs);
    const float scale = fmaxf(to_float(stored_scale), 1.0e-6f);
    const float multiplier = (Bit == 8 ? 127.0f : 7.0f) / scale;
    constexpr int value_bytes = GroupSize * Bit / 8;
    constexpr int bytes_per_group = value_bytes + sizeof(scalar_t);
    uint8_t* group_output = group_is_valid ? output + group * bytes_per_group : output;

    if (group_is_valid && Bit == 8 && sizeof(scalar_t) == 2) {
        auto* packed = reinterpret_cast<uint16_t*>(group_output);
        #pragma unroll
        for (int index = 0; index < values_per_lane; index += 2) {
            const uint16_t first = static_cast<uint16_t>(
                static_cast<uint32_t>(clamp_and_round<float>(to_float(prepared[index]) * multiplier, -127, 127)) & 0xff
            );
            const uint16_t second = static_cast<uint16_t>(
                static_cast<uint32_t>(clamp_and_round<float>(to_float(prepared[index + 1]) * multiplier, -127, 127)) & 0xff
            );
            packed[lane * (values_per_lane / 2) + index / 2] = static_cast<uint16_t>((first << 8) | second);
        }
    } else if (group_is_valid && Bit == 4 && sizeof(scalar_t) == 2) {
        auto* packed = reinterpret_cast<uint16_t*>(group_output);
        #pragma unroll
        for (int index = 0; index < values_per_lane; index += 4) {
            const uint16_t first = clamp_and_round<float>(to_float(prepared[index]) * multiplier, -7, 7) & 0x0f;
            const uint16_t second = clamp_and_round<float>(to_float(prepared[index + 1]) * multiplier, -7, 7) & 0x0f;
            const uint16_t third = clamp_and_round<float>(to_float(prepared[index + 2]) * multiplier, -7, 7) & 0x0f;
            const uint16_t fourth = clamp_and_round<float>(to_float(prepared[index + 3]) * multiplier, -7, 7) & 0x0f;
            packed[lane * (values_per_lane / 4) + index / 4] =
                static_cast<uint16_t>((first << 12) | (second << 8) | (third << 4) | fourth);
        }
    } else if (group_is_valid && Bit == 8) {
        auto* packed = reinterpret_cast<uint32_t*>(group_output);
        #pragma unroll
        for (int index = 0; index < values_per_lane; index += 4) {
            uint32_t value = 0;
            #pragma unroll
            for (int offset = 0; offset < 4; ++offset) {
                const uint32_t quantized = static_cast<uint32_t>(
                    clamp_and_round<float>(to_float(prepared[index + offset]) * multiplier, -127, 127)
                ) & 0xff;
                value = (value << 8) | quantized;
            }
            packed[lane * (values_per_lane / 4) + index / 4] = value;
        }
    } else if (group_is_valid) {
        auto* packed = reinterpret_cast<uint32_t*>(group_output);
        #pragma unroll
        for (int index = 0; index < values_per_lane; index += 8) {
            uint32_t value = 0;
            #pragma unroll
            for (int offset = 0; offset < 8; ++offset) {
                const uint32_t quantized = static_cast<uint32_t>(
                    clamp_and_round<float>(to_float(prepared[index + offset]) * multiplier, -7, 7)
                ) & 0x0f;
                value = (value << 4) | quantized;
            }
            packed[lane * (values_per_lane / 8) + index / 8] = value;
        }
    }
    if (group_is_valid && lane == 0) {
        *reinterpret_cast<scalar_t*>(group_output + value_bytes) = stored_scale;
    }
}

template <typename scalar_t, int GroupSize>
void launch_quantize_pack(
    const torch::Tensor& input,
    const c10::optional<torch::Tensor>& residual,
    torch::Tensor& output,
    int64_t bit
) {
    const int64_t num_groups = (input.numel() + GroupSize - 1) / GroupSize;
    if (num_groups == 0) {
        return;
    }
    constexpr int lanes_per_group = GroupSize / 16;
    constexpr int groups_per_block = kThreads / lanes_per_group;
    const int blocks = static_cast<int>((num_groups + groups_per_block - 1) / groups_per_block);
    const scalar_t* residual_ptr = residual.has_value()
        ? static_cast<const scalar_t*>(residual->data_ptr())
        : nullptr;
    cudaStream_t stream = get_current_cuda_stream();
    if (bit == 8) {
        quantize_pack_kernel<scalar_t, GroupSize, 8><<<blocks, kThreads, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            residual_ptr,
            static_cast<uint8_t*>(output.data_ptr()),
            input.numel()
        );
    } else {
        quantize_pack_kernel<scalar_t, GroupSize, 4><<<blocks, kThreads, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            residual_ptr,
            static_cast<uint8_t*>(output.data_ptr()),
            input.numel()
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_group_size(
    const torch::Tensor& input,
    const c10::optional<torch::Tensor>& residual,
    torch::Tensor& output,
    int64_t group_size,
    int64_t bit
) {
    if (group_size == 16) {
        launch_quantize_pack<scalar_t, 16>(input, residual, output, bit);
    } else if (group_size == 32) {
        launch_quantize_pack<scalar_t, 32>(input, residual, output, bit);
    } else {
        launch_quantize_pack<scalar_t, 64>(input, residual, output, bit);
    }
}

}  // namespace

bool inplace_quantize_pack(
    torch::Tensor input,
    torch::Tensor output,
    c10::optional<torch::Tensor> residual,
    int64_t group_size,
    int64_t topk,
    bool stochastic,
    int64_t bit,
    QuantType quant_type,
    bool compact
) {
    if (!compact || topk != 0 || stochastic || quant_type != QuantType::Linear) {
        return false;
    }
    if ((bit != 4 && bit != 8) || (group_size != 16 && group_size != 32 && group_size != 64)) {
        return false;
    }
    if (input.dtype() != torch::kHalf && input.dtype() != torch::kBFloat16 && input.dtype() != torch::kFloat32) {
        return false;
    }

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must have uint8 dtype");
    TORCH_CHECK(output.device() == input.device(), "input and output must be on the same device");
    c10::cuda::CUDAGuard device_guard(input.device());
    const int64_t num_groups = (input.numel() + group_size - 1) / group_size;
    const int64_t bytes_per_group = group_size * bit / 8 + input.element_size();
    TORCH_CHECK(output.numel() == num_groups * bytes_per_group, "output has an invalid quantized payload size");

    if (residual.has_value()) {
        TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor");
        TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
        TORCH_CHECK(residual->device() == input.device(), "input and residual must be on the same device");
        TORCH_CHECK(residual->dtype() == input.dtype(), "input and residual must have the same dtype");
        TORCH_CHECK(residual->numel() == input.numel(), "input and residual must have the same number of elements");
    }

    if (input.numel() == 0) {
        return true;
    }
    if (!residual.has_value() && input.numel() % group_size == 0) {
        inplace_quantize(input, output, group_size, topk, stochastic, bit, quant_type, compact);
        return true;
    }

    if (input.dtype() == torch::kHalf) {
        dispatch_group_size<__half>(input, residual, output, group_size, bit);
    } else if (input.dtype() == torch::kBFloat16) {
        dispatch_group_size<__nv_bfloat16>(input, residual, output, group_size, bit);
    } else {
        dispatch_group_size<float>(input, residual, output, group_size, bit);
    }
    return true;
}
