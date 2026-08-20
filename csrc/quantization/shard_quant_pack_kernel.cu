#include "quant_api.cuh"
#include "utils.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int64_t kScaleBytes = 2;

int64_t checked_mul(int64_t left, int64_t right) {
    TORCH_CHECK(left >= 0 && right >= 0, "shard quantize-pack size is invalid");
    TORCH_CHECK(
        left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
        "shard quantize-pack size overflow"
    );
    return left * right;
}

template <typename scalar_t, int GroupSize>
__global__ void shard_quantize_pack_kernel(
    const scalar_t* input,
    uint8_t* packed,
    int64_t numel,
    int64_t logical_shard_length,
    int64_t groups_per_shard
) {
    __shared__ float absolute_values[GroupSize];
    const int64_t destination =
        static_cast<int64_t>(blockIdx.x) / groups_per_shard;
    const int64_t group =
        static_cast<int64_t>(blockIdx.x) % groups_per_shard;
    const int64_t lane = threadIdx.x;
    const int64_t group_offset = group * GroupSize;
    const int64_t shard_index = group_offset + lane;
    const int64_t global_index =
        destination * logical_shard_length + shard_index;
    const bool valid =
        shard_index < logical_shard_length && global_index < numel;
    const float value = valid ? half2float(input[global_index]) : 0.0f;
    absolute_values[lane] = fabsf(value);
    __syncthreads();

    for (int offset = GroupSize / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            absolute_values[lane] = fmaxf(
                absolute_values[lane],
                absolute_values[lane + offset]
            );
        }
        __syncthreads();
    }

    constexpr int64_t bytes_per_group = GroupSize + kScaleBytes;
    const int64_t output_group =
        destination * groups_per_shard + group;
    uint8_t* group_output = packed + output_group * bytes_per_group;
    const scalar_t stored_scale =
        float2half<scalar_t>(absolute_values[0]);
    const float scale = half2float(stored_scale);
    const float multiplier = scale == 0.0f ? 0.0f : 127.0f / scale;
    const int quantized = scale == 0.0f
        ? 0
        : clamp_and_round<float>(value * multiplier, -127, 127);
    reinterpret_cast<int8_t*>(group_output + kScaleBytes)[lane] =
        static_cast<int8_t>(quantized);
    if (lane == 0) {
        *reinterpret_cast<scalar_t*>(group_output) = stored_scale;
    }
}

template <typename scalar_t, int GroupSize>
void launch_shard_quantize_pack(
    const torch::Tensor& input,
    torch::Tensor& packed,
    int64_t logical_shard_length,
    int64_t transport_shard_length,
    int64_t world_size
) {
    const int64_t groups_per_shard =
        transport_shard_length / GroupSize;
    const int64_t blocks = checked_mul(world_size, groups_per_shard);
    if (blocks == 0) {
        return;
    }
    TORCH_CHECK(
        blocks <= std::numeric_limits<int>::max(),
        "shard quantize-pack launch grid overflow"
    );
    cudaStream_t stream = get_current_cuda_stream();
    shard_quantize_pack_kernel<scalar_t, GroupSize>
        <<<static_cast<int>(blocks), GroupSize, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<uint8_t*>(packed.data_ptr()),
            input.numel(),
            logical_shard_length,
            groups_per_shard
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_group_size(
    const torch::Tensor& input,
    torch::Tensor& packed,
    int64_t logical_shard_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size
) {
    if (group_size == 16) {
        launch_shard_quantize_pack<scalar_t, 16>(
            input,
            packed,
            logical_shard_length,
            transport_shard_length,
            world_size
        );
    } else if (group_size == 32) {
        launch_shard_quantize_pack<scalar_t, 32>(
            input,
            packed,
            logical_shard_length,
            transport_shard_length,
            world_size
        );
    } else {
        launch_shard_quantize_pack<scalar_t, 64>(
            input,
            packed,
            logical_shard_length,
            transport_shard_length,
            world_size
        );
    }
}

bool private_shard_quantize_pack(
    const torch::Tensor& input,
    torch::Tensor packed,
    int64_t logical_shard_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size
) {
    return try_inplace_shard_quantize_pack(
        input,
        packed,
        logical_shard_length,
        transport_shard_length,
        world_size,
        group_size
    );
}

}  // namespace

bool try_inplace_shard_quantize_pack(
    const torch::Tensor& input,
    torch::Tensor& packed,
    int64_t logical_shard_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size
) {
    if (input.dtype() != torch::kHalf &&
        input.dtype() != torch::kBFloat16) {
        return false;
    }
    if (group_size != 16 && group_size != 32 && group_size != 64) {
        return false;
    }
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(
        packed.dtype() == torch::kUInt8,
        "packed must have uint8 dtype"
    );
    TORCH_CHECK(
        packed.device() == input.device(),
        "input and packed must be on the same device"
    );
    TORCH_CHECK(packed.dim() == 1, "packed must be one-dimensional");
    TORCH_CHECK(world_size > 0, "world size must be positive");
    TORCH_CHECK(
        logical_shard_length >= 0,
        "logical shard length must be non-negative"
    );
    TORCH_CHECK(
        transport_shard_length >= 0,
        "transport shard length must be non-negative"
    );

    const int64_t numel = input.numel();
    const int64_t expected_logical =
        numel / world_size + (numel % world_size != 0);
    TORCH_CHECK(
        logical_shard_length == expected_logical,
        "logical shard length does not match the input layout"
    );
    const int64_t groups_per_shard =
        logical_shard_length / group_size +
        (logical_shard_length % group_size != 0);
    const int64_t expected_transport = checked_mul(
        groups_per_shard,
        group_size
    );
    TORCH_CHECK(
        transport_shard_length == expected_transport,
        "transport shard length does not match the input layout"
    );
    const int64_t bytes_per_group = group_size + kScaleBytes;
    const int64_t expected_payload = checked_mul(
        checked_mul(world_size, groups_per_shard),
        bytes_per_group
    );
    TORCH_CHECK(
        packed.numel() == expected_payload,
        "packed payload size does not match the shard layout"
    );

    c10::cuda::CUDAGuard device_guard(input.device());
    if (input.dtype() == torch::kHalf) {
        dispatch_group_size<__half>(
            input,
            packed,
            logical_shard_length,
            transport_shard_length,
            world_size,
            group_size
        );
    } else {
        dispatch_group_size<__nv_bfloat16>(
            input,
            packed,
            logical_shard_length,
            transport_shard_length,
            world_size,
            group_size
        );
    }
    return true;
}

TORCH_LIBRARY_FRAGMENT(lowbit_comm_private, module) {
    module.def(
        "shard_quantize_pack(Tensor input, Tensor(a!) packed, "
        "int logical_shard_length, int transport_shard_length, "
        "int world_size, int group_size) -> bool"
    );
}

TORCH_LIBRARY_IMPL(lowbit_comm_private, CUDA, module) {
    module.impl(
        "shard_quantize_pack",
        TORCH_FN(private_shard_quantize_pack)
    );
}
