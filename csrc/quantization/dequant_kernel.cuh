#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "utils.cuh"
#include "inner.cuh"
#include "enum.cuh"


template <typename scalar_t, int GroupSize, int ThreadsPerGroup, int Bit, QuantType Type>
__device__ void dequant_loop(uint16_t* srd, float scale) {
    int64_t st = threadIdx.x * GroupSize / ThreadsPerGroup;
    const int64_t num_bank_per_thread = (GroupSize / ThreadsPerGroup) / 4;
    const int64_t start_bank_id = (threadIdx.x & 31) / (32 / num_bank_per_thread);
    const int64_t i_st = start_bank_id * 4;

    if constexpr (Type == QuantType::Linear) {
        scale = scale / (Bit == 8 ? 127.0 : 7.0);
    }

    uint16_t raw_data = 0;
    for (int64_t __i = GroupSize / ThreadsPerGroup - 1, _i, i; __i >= 0; --__i) {
        i = (i_st + __i) & (GroupSize / ThreadsPerGroup - 1);
        _i = i & (Bit == 8 ? 1 : 3);
        float value;
        if constexpr (Bit == 8) {
            raw_data >>= 8;
            if ((i & 1) == 1)
                raw_data = srd[st + i - 1];
            value = dequant_inner<scalar_t, 8, Type>(raw_data & 255, scale);
        } else {
            raw_data >>= 4;
            if ((i & 3) == 3)
                raw_data = srd[st + i - 3];
            value = dequant_inner<scalar_t, 4, Type>(raw_data & 15, scale);
        }
        ((scalar_t*)srd)[st + i] = float2half<scalar_t>(value);
    }
} 

template <ReduceOP OP, typename scalar_t>
__device__ void shared_to_global_op(scalar_t* srd, scalar_t* glb, int64_t len) {
    int64_t step = blockDim.x;
    for (int64_t index = threadIdx.x; index < len; index += step) {
        if constexpr (OP == ReduceOP::SUM) {
            glb[index] += srd[index];
        } else if constexpr (OP == ReduceOP::MAX) {
            glb[index] = hfmax(glb[index], srd[index]);
        } else if constexpr (OP == ReduceOP::MIN) {
            glb[index] = hfmin(glb[index], srd[index]);
        } else {
            glb[index] = srd[index];
        }
    }
}

template <typename scalar_t, int GroupSize, int TopK, int ThreadsPerGroup, int Bit, ReduceOP OP, QuantType Type, bool Even>
__global__ void dequant_kernel(uint16_t* input, uint16_t* output, int64_t length) {
    static_assert(sizeof(uint16_t) == sizeof(scalar_t), "Not Support Dtype.");
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 2 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 4)), "GroupSize should be multiple of 2 for 8bit quantization or multiple of 4 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");

    extern __shared__ uint16_t shared[];

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    if constexpr (Even) {
        if constexpr (Bit==8) {
            global_to_shared<2>(input + block_st_index / 2, shared, num_group_per_block * GroupSize / 2);
        } else {
            global_to_shared<4>(input + block_st_index / 4, shared, num_group_per_block * GroupSize / 4);
        }
    } else {
        if (block_st_index + num_group_per_block * GroupSize > length) {
            if constexpr (Bit==8) {
                global_to_shared<2>(input + block_st_index / 2, shared, (length - block_st_index) / 2);
            } else {
                global_to_shared<4>(input + block_st_index / 4, shared, (length - block_st_index) / 4);
            }
        } else {
            if constexpr (Bit==8) {
                global_to_shared<2>(input + block_st_index / 2, shared, num_group_per_block * GroupSize / 2);
            } else {
                global_to_shared<4>(input + block_st_index / 4, shared, num_group_per_block * GroupSize / 4);
            }
        }
    }
    
    __syncthreads();

    // load scale 
    int64_t scale_index;
    if constexpr (Even) 
        scale_index = num_group_per_block * gridDim.x * GroupSize / (Bit == 8 ? 2 : 4);
    else
        scale_index = length / (Bit == 8 ? 2 : 4);
    
    if constexpr (TopK == 0)
        scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup));
    else if constexpr (TopK == 1) 
        scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 3;
    else if constexpr (TopK == 2) 
        scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 4;

    if constexpr (Even) {
        scalar_t scale = ((scalar_t*)input)[scale_index];
        dequant_loop<scalar_t, GroupSize, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(scale));
    } else {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup <= length) {
            scalar_t scale = ((scalar_t*)input)[scale_index];
            dequant_loop<scalar_t, GroupSize, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(scale));
        }
    }

    // load topk
    bool isMainThread = ((threadIdx.x & (ThreadsPerGroup - 1)) == 0);
    if constexpr (!Even) {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup > length)
            isMainThread = false;
    }
    if (isMainThread) {
        if constexpr (TopK == 1) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            uint16_t top_index = input[scale_index + 1];
            scalar_t top_value = ((scalar_t*)input)[scale_index + 2];
            assert (top_index < GroupSize);
            ((scalar_t*)shared)[group_st_index + top_index] = top_value;
        } else if constexpr (TopK == 2) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            uint16_t top_index = input[scale_index + 1];
            uint16_t top_index1 = top_index & 0x00FF;
            uint16_t top_index2 = top_index >> 8;
            scalar_t top_value1 = ((scalar_t*)input)[scale_index + 2];
            scalar_t top_value2 = ((scalar_t*)input)[scale_index + 3];
            assert (top_index1 < GroupSize);
            assert (top_index2 < GroupSize);
            ((scalar_t*)shared)[group_st_index + top_index1] = top_value1;
            ((scalar_t*)shared)[group_st_index + top_index2] = top_value2;
        }
    }

    __syncthreads();

    // save to output
    if constexpr (Even) {
        shared_to_global_op<OP, scalar_t>((scalar_t*)shared, (scalar_t*)(output + block_st_index), num_group_per_block * GroupSize);
    } else {
        if (block_st_index + num_group_per_block * GroupSize > length)
            shared_to_global_op<OP, scalar_t>((scalar_t*)shared, (scalar_t*)(output + block_st_index), (length - block_st_index));
        else
            shared_to_global_op<OP, scalar_t>((scalar_t*)shared, (scalar_t*)(output + block_st_index), num_group_per_block * GroupSize);
    }
}   

template <typename scalar_t, int GroupSize, int TopK, int ThreadsPerGroup, int Bit, ReduceOP OP, QuantType Type, bool Even>
__global__ void dequant_kernel_compact(uint16_t* input, uint16_t* output, int64_t length) {
    static_assert(sizeof(uint16_t) == sizeof(scalar_t), "Not Support Dtype.");
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 2 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 4)), "GroupSize should be multiple of 2 for 8bit quantization or multiple of 4 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");
    static_assert(ThreadsPerGroup==1, "Only Support ThreadsPerGroup == 1.");

    extern __shared__ uint16_t shared[];

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    {
        u_int16_t buffer[8];
        int64_t element_per_group = GroupSize / (Bit == 4 ? 4 : 2) + 1;
        if constexpr (TopK == 1) element_per_group += 2;
        else if constexpr (TopK == 2) element_per_group += 3;
        int64_t total_element = element_per_group * num_group_per_block;
        int64_t global_st = total_element * blockIdx.x;
        if constexpr (!Even) {
            total_element = min(total_element, ((length - block_st_index) / GroupSize) * element_per_group);
        }
        for (int64_t i = threadIdx.x * 8, group_id, group_offset; i < total_element; i += 8 * blockDim.x) {
            if (i + 8 > total_element) {
                for (int64_t j = 0; j < total_element - i; ++j) {
                    group_id = (i + j) / element_per_group;
                    group_offset = (i + j) - group_id * element_per_group;
                    *(shared + GroupSize * group_id + group_offset) = *(input + global_st + i + j);
                }
            } else {
                *((int4*)buffer) = *((int4 *)(input + global_st + i));
                for (int64_t j = 0; j < 8; ++j) {
                    group_id = (i + j) / element_per_group;
                    group_offset = (i + j) - group_id * element_per_group;
                    *(shared + GroupSize * group_id + group_offset) = buffer[j];
                }
            }
        }
    }

    __syncthreads();

    // rearrange shared memory, load scale and topk
    scalar_t scale, top1, top2;
    uint16_t top1_index, top2_index;
    {
        if constexpr (Even) {
            int64_t group_st = threadIdx.x * GroupSize;
            int64_t num_element = GroupSize / (Bit == 4 ? 4 : 2);
            scale = ((scalar_t*)shared)[group_st + num_element];
            if constexpr (TopK == 1) {
                top1_index = shared[group_st + num_element + 1];
                top1 = ((scalar_t*)shared)[group_st + num_element + 2];
            } else if constexpr (TopK == 2) {
                top1_index = shared[group_st + num_element + 1];
                top1 = ((scalar_t*)shared)[group_st + num_element + 2];
                top2 = ((scalar_t*)shared)[group_st + num_element + 3];
                top2_index = (top1_index >> 8) & 255;
                top1_index = top1_index & 255;
            }
            for (int i = num_element - 1; i; --i) {
                shared[group_st + i * (Bit == 4 ? 4 : 2)] = shared[group_st + i];
            }
        } else {
            if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= length) {
                int64_t group_st = threadIdx.x * GroupSize;
                int64_t num_element = GroupSize / (Bit == 4 ? 4 : 2);
                scale = ((scalar_t*)shared)[group_st + num_element];
                if constexpr (TopK == 1) {
                    top1_index = shared[group_st + num_element + 1];
                    top1 = ((scalar_t*)shared)[group_st + num_element + 2];
                } else if constexpr (TopK == 2) {
                    top1_index = shared[group_st + num_element + 1];
                    top1 = ((scalar_t*)shared)[group_st + num_element + 2];
                    top2 = ((scalar_t*)shared)[group_st + num_element + 3];
                    top2_index = (top1_index >> 8) & 255;
                    top1_index = top1_index & 255;
                }
                for (int i = num_element - 1; i; --i) {
                    shared[group_st + i * (Bit == 4 ? 4 : 2)] = shared[group_st + i];
                }
            }
        }
    }

    // dequant
    if constexpr (Even) {
        dequant_loop<scalar_t, GroupSize, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(scale));
    } else {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup <= length) {
            dequant_loop<scalar_t, GroupSize, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(scale));
        }
    }

    // load topk
    bool isMainThread = true;
    if constexpr (!Even) {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup > length)
            isMainThread = false;
    }
    if (isMainThread) {
        if constexpr (TopK == 1) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            assert (top1_index < GroupSize);
            ((scalar_t*)shared)[group_st_index + top1_index] = top1;
        } else if constexpr (TopK == 2) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            assert (top1_index < GroupSize);
            assert (top2_index < GroupSize);
            ((scalar_t*)shared)[group_st_index + top1_index] = top1;
            ((scalar_t*)shared)[group_st_index + top2_index] = top2;
        }
    }

    __syncthreads();

    // save to output
    if constexpr (Even) {
        shared_to_global_op<OP, scalar_t>((scalar_t*)shared, (scalar_t*)(output + block_st_index), num_group_per_block * GroupSize);
    } else {
        if (block_st_index + num_group_per_block * GroupSize > length)
            shared_to_global_op<OP, scalar_t>((scalar_t*)shared, (scalar_t*)(output + block_st_index), (length - block_st_index));
        else
            shared_to_global_op<OP, scalar_t>((scalar_t*)shared, (scalar_t*)(output + block_st_index), num_group_per_block * GroupSize);
    }
}   