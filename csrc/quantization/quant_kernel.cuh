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



template <typename scalar_t> 
struct TopKRet {
    scalar_t top1;
    scalar_t top2;
    scalar_t scale;
    int top1_index;
    int top2_index;
};

template <typename scalar_t, int TopK, int GroupSize, int ThreadsPerGroup>
__device__ void get_topk_and_scale(scalar_t* srd, TopKRet<scalar_t>& ret) {
    static_assert(sizeof(scalar_t) == sizeof(uint16_t), "Unsupported Dtype.");
    static_assert((GroupSize / ThreadsPerGroup) % 4 == 0, "GroupSize should be multiple of 4.");
    static_assert((GroupSize / ThreadsPerGroup) <= 128, "GroupSize should be less than 128.");
    const int64_t num_bank_per_thread = (GroupSize / ThreadsPerGroup) / 4;
    const int64_t start_bank_id = (threadIdx.x & 31) / (32 / num_bank_per_thread);
    const int64_t i_st = start_bank_id * 4;
    if constexpr (ThreadsPerGroup == 1) {
        if constexpr (TopK == 0) {
            scalar_t scale = float2half<scalar_t>(0.0);
            for (int64_t i = 0, st = threadIdx.x * GroupSize; i < GroupSize; ++i)
                scale = hfmax(scale, __habs(srd[st + ((i + i_st) & (GroupSize - 1))]));
            ret.scale = scale;
        } else if constexpr (TopK == 1) {
            scalar_t 
                top1 = float2half<scalar_t>(0.0), 
                top2 = float2half<scalar_t>(0.0);
            int64_t top1_index = -1;
            const int64_t st = threadIdx.x * GroupSize;
            for (int64_t _i = 0, i; _i < GroupSize; ++_i) {
                i = (_i + i_st) & (GroupSize - 1);
                scalar_t tmp = __habs(srd[st + i]);
                if (__hgt(tmp, top1)) {
                    top2 = top1;
                    top1_index = i;
                    top1 = tmp;
                } else if (__hgt(tmp, top2))
                    top2 = tmp;
            }
            if (top1_index == -1) top1_index = 0, top2 = __habs(srd[st + 1]);
            ret.top1 = srd[st + top1_index];
            ret.top1_index = top1_index;
            ret.scale = top2;
            srd[st + top1_index] = float2half<scalar_t>(0.0);

        } else if constexpr (TopK == 2) {
            scalar_t 
                top1 = float2half<scalar_t>(0.0), 
                top2 = float2half<scalar_t>(0.0), 
                top3 = float2half<scalar_t>(0.0);
            int64_t top1_index = -1, top2_index = -1;
            const int64_t st = threadIdx.x * GroupSize;
            for (int64_t _i = 0, i; _i < GroupSize; ++_i) {
                i = (_i + i_st) & (GroupSize - 1);
                scalar_t tmp = __habs(srd[st + i]);
                if (__hgt(tmp, top1)) {
                    top3 = top2;
                    top2 = top1;
                    top2_index = top1_index;
                    top1 = tmp;
                    top1_index = i;
                } else if (__hgt(tmp, top2)) {
                    top3 = top2;
                    top2 = tmp;
                    top2_index = i;
                } else if (__hgt(tmp, top3))
                    top3 = tmp;
            }
            if (top1_index == -1) { // all zero/nan
                top1_index = 0;
                top2_index = 1;
                top3 = float2half<scalar_t>(0.0);
            } else if (top2_index == -1) { // all the same or only one non-zero/non-nan
                top2_index = (top1_index + 1) % GroupSize;
                top3 = float2half<scalar_t>(0.0);
            }
            ret.top1 = srd[st + top1_index];
            ret.top2 = srd[st + top2_index];
            ret.scale = top3;
            ret.top1_index = top1_index;
            ret.top2_index = top2_index;
            srd[st + top1_index] = float2half<scalar_t>(0.0);
            srd[st + top2_index] = float2half<scalar_t>(0.0);
        } else {
            // Not Implemented
            static_assert(TopK <= 2, "Not Implemented");
        }
    } else {
        // Not Implemented
        static_assert(ThreadsPerGroup==1, "Not Implemented");
    }

    return;
}

template <typename scalar_t, int GroupSize, bool Stochastic, int ThreadsPerGroup, int Bit, QuantType Type>
__device__ void quant_loop(uint16_t* shared, float scale, std::pair<uint64_t, uint64_t> seed) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    if (scale < 1e-6) scale = 1e-6;
    const int64_t num_bank_per_thread = (GroupSize / ThreadsPerGroup) / 4;
    const int64_t start_bank_id = (threadIdx.x & 31) / (32 / num_bank_per_thread);
    const int64_t i_st = start_bank_id * 4;
    if constexpr (Stochastic) {
        curand_init(seed.first, index, seed.second, &state);
    }

    if constexpr (Type == QuantType::Linear) {
        scale = (Bit == 4 ? 7.0 : 127.0) / scale;
    }

    if constexpr (ThreadsPerGroup == 1) {
        uint16_t value = 0, tmp;
        for (int64_t _i = 0, st = threadIdx.x * GroupSize, i; _i < GroupSize; ++_i) {
            i = (_i + i_st) & (GroupSize - 1);
            tmp = quant_inner<scalar_t, Stochastic, Bit, Type>(((scalar_t*)shared)[st+i], scale, state);
            if constexpr (Bit == 8) {
                value <<= 8;
                if (i&1) {
                    shared[st+i-1] = tmp | value;
                    value = 0;
                } else {
                    value = tmp;
                }
            } else if constexpr (Bit == 4) {
                value <<= 4;
                value |= tmp;
                if ((i&3) == 3) {
                    shared[st+i-3] = value;
                    value = 0;
                }
            }
        }

    } else {
        static_assert(ThreadsPerGroup == 1, "Not Implemented.");
    }
}


template <typename scalar_t, int GroupSize, int TopK, bool Stochastic, int ThreadsPerGroup, int Bit, QuantType Type, bool Even>
__global__ void quant_kernel(uint16_t* input, uint16_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len) {
    static_assert(sizeof(uint16_t) == sizeof(scalar_t), "Not Support Dtype.");
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 2 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 4)), "GroupSize should be multiple of 2 for 8bit quantization or multiple of 4 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");
    static_assert(GroupSize <= 256, "GroupSize should be less than 256.");

    extern __shared__ uint16_t shared[];

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    if constexpr (Even) 
        global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
    else {
        if (block_st_index + num_group_per_block * GroupSize <= input_len)
            global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
        else
            global_to_shared<1>(input + block_st_index, shared, input_len - block_st_index);
    }

    __syncthreads();

    // get max abs value
    TopKRet<scalar_t> topk_ret;
    if constexpr (Even)
        get_topk_and_scale<scalar_t, TopK, GroupSize, ThreadsPerGroup>((scalar_t*)shared, topk_ret);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            get_topk_and_scale<scalar_t, TopK, GroupSize, ThreadsPerGroup>((scalar_t*)shared, topk_ret);
        else {
            if constexpr (ThreadsPerGroup > 1) {
                __syncthreads(); // get_topk_and_scale has __syncthreads() inside when ThreadsPerGroup > 1
            }   
        }
    }

    // quantization
    if constexpr (Even) 
        quant_loop<scalar_t, GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(topk_ret.scale), seed);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            quant_loop<scalar_t, GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(topk_ret.scale), seed);
    }

    bool isMainThread = ((threadIdx.x & (ThreadsPerGroup - 1)) == 0);

    if constexpr (!Even) {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) > input_len)
            isMainThread = false;
    }

    __syncthreads();

    // save to global memory
    if constexpr (Even) {
        if constexpr (Bit==8)
            shared_to_global<2>(shared, output + (block_st_index / 2), num_group_per_block * GroupSize / 2);
        else
            shared_to_global<4>(shared, output + (block_st_index / 4), num_group_per_block * GroupSize / 4);
    } else {
        if (block_st_index + num_group_per_block * GroupSize <= input_len) {
            if constexpr (Bit==8)
                shared_to_global<2>(shared, output + (block_st_index / 2), num_group_per_block * GroupSize / 2);
            else
                shared_to_global<4>(shared, output + (block_st_index / 4), num_group_per_block * GroupSize / 4);
        } else {
            if constexpr (Bit==8)
                shared_to_global<2>(shared, output + (block_st_index / 2), (input_len - block_st_index) / 2);
            else
                shared_to_global<4>(shared, output + (block_st_index / 4), (input_len - block_st_index) / 4);
        }
    }

    // save scale and topk
    int64_t scale_index;
    if (isMainThread) {
        if constexpr (Bit == 8) {
            if constexpr (Even)
                scale_index = num_group_per_block * gridDim.x * GroupSize / 2;
            else
                scale_index = input_len / 2;

            if constexpr (TopK == 0)
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup));
            else if constexpr (TopK == 1) 
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 3;
            else if constexpr (TopK == 2) 
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 4;


            *((scalar_t*)(output + scale_index)) = topk_ret.scale;
            if constexpr (TopK == 1) {
                output[scale_index + 1] = topk_ret.top1_index;
                ((scalar_t*)output)[scale_index + 2] = topk_ret.top1;
            } else if constexpr (TopK == 2) {
                output[scale_index + 1] = (topk_ret.top1_index | (topk_ret.top2_index << 8));
                ((scalar_t*)output)[scale_index + 2] = topk_ret.top1;
                ((scalar_t*)output)[scale_index + 3] = topk_ret.top2;
            }

        } else if constexpr (Bit == 4) {
            if constexpr (Even)
                scale_index = num_group_per_block * gridDim.x * GroupSize / 4;
            else
                scale_index = input_len / 4;

            if constexpr (TopK == 0)
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup));
            else if constexpr (TopK == 1) 
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 3;
            else if constexpr (TopK == 2) 
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 4;


            *((scalar_t*)(output + scale_index)) = topk_ret.scale;
            if constexpr (TopK == 1) {
                output[scale_index + 1] = topk_ret.top1_index;
                ((scalar_t*)output)[scale_index + 2] = topk_ret.top1;
            } else if constexpr (TopK == 2) {
                output[scale_index + 1] = (topk_ret.top1_index | (topk_ret.top2_index << 8));
                ((scalar_t*)output)[scale_index + 2] = topk_ret.top1;
                ((scalar_t*)output)[scale_index + 3] = topk_ret.top2;
            }
        }
    }
}


template <typename scalar_t, int GroupSize, int TopK, bool Stochastic, int ThreadsPerGroup, int Bit, QuantType Type, bool Even>
__global__ void quant_kernel_compact(uint16_t* input, uint16_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len) {
    static_assert(sizeof(uint16_t) == sizeof(scalar_t), "Not Support Dtype.");
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 2 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 4)), "GroupSize should be multiple of 2 for 8bit quantization or multiple of 4 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");
    static_assert(GroupSize <= 256, "GroupSize should be less than 256.");

    extern __shared__ uint16_t shared[];

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    if constexpr (Even) 
        global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
    else {
        if (block_st_index + num_group_per_block * GroupSize <= input_len)
            global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
        else
            global_to_shared<1>(input + block_st_index, shared, input_len - block_st_index);
    }

    __syncthreads();

    // get max abs value
    TopKRet<scalar_t> topk_ret;
    if constexpr (Even)
        get_topk_and_scale<scalar_t, TopK, GroupSize, ThreadsPerGroup>((scalar_t*)shared, topk_ret);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            get_topk_and_scale<scalar_t, TopK, GroupSize, ThreadsPerGroup>((scalar_t*)shared, topk_ret);
        else {
            if constexpr (ThreadsPerGroup > 1) {
                __syncthreads(); // get_topk_and_scale has __syncthreads() inside when ThreadsPerGroup > 1
            }   
        }
    }

    // quantization
    if constexpr (Even) 
        quant_loop<scalar_t, GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(topk_ret.scale), seed);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            quant_loop<scalar_t, GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, half2float<scalar_t>(topk_ret.scale), seed);
    }


    // rearrange shared memory
    if constexpr (Even) {
        int64_t group_st = threadIdx.x * GroupSize;
        int64_t num_element = GroupSize / (Bit == 4 ? 4 : 2);
        for (int i = 1; i < num_element; ++i)
            shared[group_st + i] = shared[group_st + i * (Bit == 4 ? 4 : 2)];

        ((scalar_t*)shared)[group_st + num_element] = topk_ret.scale;
        if constexpr (TopK == 1) {
            shared[group_st + num_element + 1] = topk_ret.top1_index;
            ((scalar_t*)shared)[group_st + num_element + 2] = topk_ret.top1;
        } else if constexpr (TopK == 2) {
            shared[group_st + num_element + 1] = topk_ret.top1_index | (topk_ret.top2_index << 8);
            ((scalar_t*)shared)[group_st + num_element + 2] = topk_ret.top1;
            ((scalar_t*)shared)[group_st + num_element + 3] = topk_ret.top2;
        }
    } else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len) {
            int64_t group_st = threadIdx.x * GroupSize;
            int64_t num_element = GroupSize / (Bit == 4 ? 4 : 2);
            for (int i = 1; i < num_element; ++i)
                shared[group_st + i] = shared[group_st + i * (Bit == 4 ? 4 : 2)];

            ((scalar_t*)shared)[group_st + num_element] = topk_ret.scale;
            if constexpr (TopK == 1) {
                shared[group_st + num_element + 1] = topk_ret.top1_index;
                ((scalar_t*)shared)[group_st + num_element + 2] = topk_ret.top1;
            } else if constexpr (TopK == 2) {
                shared[group_st + num_element + 1] = topk_ret.top1_index | (topk_ret.top2_index << 8);
                ((scalar_t*)shared)[group_st + num_element + 2] = topk_ret.top1;
                ((scalar_t*)shared)[group_st + num_element + 3] = topk_ret.top2;
            }
        }
    }

    
    __syncthreads();

    // save to global memory
    u_int16_t buffer[8];
    int64_t element_per_group = GroupSize / (Bit == 4 ? 4 : 2) + 1;
    if constexpr (TopK == 1) element_per_group += 2;
    else if constexpr (TopK == 2) element_per_group += 3;
    int64_t total_element = element_per_group * num_group_per_block;
    int64_t global_st = total_element * blockIdx.x;
    if constexpr (!Even) {
        total_element = min(total_element, ((input_len - num_group_per_block * blockIdx.x * GroupSize) / GroupSize) * element_per_group);
    }
    for (int64_t i = threadIdx.x * 8, group_id, group_offset; i < total_element; i += 8 * blockDim.x) {
        if (i + 8 > total_element) {
            for (int64_t j = 0; j < total_element - i; ++j) {
                group_id = (i + j) / element_per_group;
                group_offset = (i + j) - group_id * element_per_group;
                *(output + global_st + i + j) = *(shared + GroupSize * group_id + group_offset);
            }
        } else {
            for (int64_t j = 0; j < 8; ++j) {
                group_id = (i + j) / element_per_group;
                group_offset = (i + j) - group_id * element_per_group;
                buffer[j] = *(shared + GroupSize * group_id + group_offset);
            }
            *((int4*)(output + global_st + i)) = *((int4*)buffer);
        }
    }
}