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



namespace quant_fp32 {
template <int64_t Step>
__device__ void shared_to_global(uint32_t* srd, uint32_t* glb, int64_t len) {
    uint32_t buffer[4];
    int64_t save_per_thread = (len & (-512)) / 128;
    int64_t st = save_per_thread * threadIdx.x;
    for (int64_t index = 0; index < save_per_thread; index += 4) {
        #pragma unroll
        for (int64_t j = 0; j < 4; ++j)
            buffer[j] = srd[(st + index + j) * Step];
        *((int4*)(glb + st + index)) = *((int4*)buffer);
    }
    if (threadIdx.x < 32) {
        int64_t remain = len - (len & (-512));
        if (remain > 0) {
            for (int64_t index = threadIdx.x * 4 + (len & (-512)); index < len; index += 128) {
                if (index + 4 <= len) {
                    #pragma unroll
                    for (int64_t j = 0; j < 4; ++j)
                        buffer[j] = srd[(index + j) * Step];
                    *((int4*)(glb + index)) = *((int4*)buffer);
                } else {
                    for (int64_t j = 0; j < 4; ++j) {
                        if (index + j < len)
                            glb[index + j] = srd[(index + j) * Step];
                    }
                }
            }
        }
    }
}

template <int64_t Step>
__device__ void global_to_shared(uint32_t* glb, uint32_t* srd, int64_t len) {
    uint32_t buffer[4];
    int64_t load_per_thread = (len & (-512)) / 128;
    int64_t st = load_per_thread * threadIdx.x;
    for (int64_t index = 0; index < load_per_thread; index += 4) {
        *((int4*)buffer) = *((int4*)(glb + st + index));
        #pragma unroll
        for (int64_t j = 0; j < 4; ++j)
            srd[(st + index + j) * Step] = buffer[j];
    }
    if (threadIdx.x < 32) { // the first warp
        int64_t remain = len - (len & (-512));
        if (remain > 0) {
            for (int64_t index = threadIdx.x * 4 + (len & (-512)); index < len; index += 128) {
                if (index + 4 <= len) {
                    *((int4*)buffer) = *((int4*)(glb + index));
                    #pragma unroll
                    for (int64_t j = 0; j < 4; ++j)
                        srd[(index + j) * Step] = buffer[j];
                } else {
                    for (int64_t j = 0; j < 4; ++j) {
                        if (index + j < len)
                            srd[(index + j) * Step] = glb[(index + j) * Step];
                    }
                }
            }
        }
    }
}


struct TopKRet {
    float top1;
    float top2;
    float scale;
    int top1_index;
    int top2_index;
};

template <int TopK, int GroupSize, int ThreadsPerGroup>
__device__ void get_topk_and_scale(float* srd, TopKRet& ret) {
    static_assert((GroupSize / ThreadsPerGroup) % 4 == 0, "GroupSize should be multiple of 4.");
    static_assert((GroupSize / ThreadsPerGroup) <= 128, "GroupSize should be less than 128.");
    if constexpr (ThreadsPerGroup == 1) {
        if constexpr (TopK == 0) {
            float scale = 0.0;
            for (int64_t i = 0, st = threadIdx.x * GroupSize; i < GroupSize; ++i)
                scale = fmax(scale, fabs(srd[st + i]));
            ret.scale = scale;
        } else if constexpr (TopK == 1) {
            float 
                top1 = 0.0, 
                top2 = 0.0;
            int64_t top1_index = -1;
            const int64_t st = threadIdx.x * GroupSize;
            for (int64_t i = 0; i < GroupSize; ++i) {
                float tmp = fabs(srd[st + i]);
                if (tmp >= top1) {
                    top2 = top1;
                    top1_index = i;
                    top1 = tmp;
                } else if (tmp >= top2)
                    top2 = tmp;
            }
            if (top1_index == -1) top1_index = 0, top2 = srd[st + 1];
            ret.top1 = srd[st + top1_index];
            ret.top1_index = top1_index;
            ret.scale = top2;
            srd[st + top1_index] = 0.0;

        } else if constexpr (TopK == 2) {
            float 
                top1 = 0.0, 
                top2 = 0.0, 
                top3 = 0.0;
            int64_t top1_index = -1, top2_index = -1;
            const int64_t st = threadIdx.x * GroupSize;
            for (int64_t i = 0; i < GroupSize; ++i) {
                float tmp = fabs(srd[st + i]);
                if (tmp >= top1) {
                    top3 = top2;
                    top2 = top1;
                    top2_index = top1_index;
                    top1 = tmp;
                    top1_index = i;
                } else if (tmp >= top2) {
                    top3 = top2;
                    top2 = tmp;
                    top2_index = i;
                } else if (tmp >= top3)
                    top3 = tmp;
            }
            if (top1_index == -1) { // all zero/nan
                top1_index = 0;
                top2_index = 1;
                top3 = 0.0;
            } else if (top2_index == -1) { // all the same or only one non-zero/non-nan
                top2_index = (top1_index + 1) % GroupSize;
                top3 = 0.0;
            }
            ret.top1 = srd[st + top1_index];
            ret.top2 = srd[st + top2_index];
            ret.scale = top3;
            ret.top1_index = top1_index;
            ret.top2_index = top2_index;
            srd[st + top1_index] = 0.0;
            srd[st + top2_index] = 0.0;
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

template <int GroupSize, bool Stochastic, int ThreadsPerGroup, int Bit, QuantType Type>
__device__ void quant_loop(uint32_t* shared, float scale, std::pair<uint64_t, uint64_t> seed) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    if (scale < 1e-6) scale = 1e-6;
    if constexpr (Stochastic) {
        curand_init(seed.first, index, seed.second, &state);
    }

    if constexpr (Type == QuantType::Linear) {
        scale = (Bit == 4 ? 7.0 : 127.0) / scale;
    }

    if constexpr (ThreadsPerGroup == 1) {
        uint32_t value = 0, tmp;
        for (int64_t i = 0, st = threadIdx.x * GroupSize; i < GroupSize; ++i) {
            tmp = quant_inner<float, Stochastic, Bit, Type>(((float*)shared)[st+i], scale, state);
            if constexpr (Bit == 8) {
                value <<= 8;
                value |= tmp;
                if ((i&3) == 3) {
                    shared[st+i-3] = tmp | value;
                    value = 0;
                }
            } else if constexpr (Bit == 4) {
                value <<= 4;
                value |= tmp;
                if ((i&7) == 7) {
                    shared[st+i-7] = value;
                    value = 0;
                }
            }
        }

    } else {
        static_assert(ThreadsPerGroup == 1, "Not Implemented.");
    }
}


}

template <int GroupSize, int TopK, bool Stochastic, int ThreadsPerGroup, int Bit, QuantType Type, bool Even>
__global__ void quant_kernel_fp32(uint32_t* input, uint32_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len) {
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 8 == 0) && (Bit == 4)), "GroupSize should be multiple of 4 for 8bit quantization or multiple of 8 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");
    static_assert(GroupSize <= 256, "GroupSize should be less than 256.");

    extern __shared__ uint16_t _shared[];
    uint32_t* shared = (uint32_t*)_shared;

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    if constexpr (Even) 
        quant_fp32::global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
    else {
        if (block_st_index + num_group_per_block * GroupSize <= input_len)
            quant_fp32::global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
        else
            quant_fp32::global_to_shared<1>(input + block_st_index, shared, input_len - block_st_index);
    }

    __syncthreads();

    // get max abs value
    quant_fp32::TopKRet  topk_ret;
    if constexpr (Even)
        quant_fp32::get_topk_and_scale<TopK, GroupSize, ThreadsPerGroup>((float*)shared, topk_ret);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            quant_fp32::get_topk_and_scale<TopK, GroupSize, ThreadsPerGroup>((float*)shared, topk_ret);
        else {
            if constexpr (ThreadsPerGroup > 1) {
                __syncthreads(); // get_topk_and_scale has __syncthreads() inside when ThreadsPerGroup > 1
            }   
        }
    }

    // quantization
    if constexpr (Even) 
        quant_fp32::quant_loop<GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, topk_ret.scale, seed);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            quant_fp32::quant_loop<GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, topk_ret.scale, seed);
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
            quant_fp32::shared_to_global<4>(shared, output + (block_st_index / 4), num_group_per_block * GroupSize / 4);
        else
            quant_fp32::shared_to_global<8>(shared, output + (block_st_index / 8), num_group_per_block * GroupSize / 8);
    } else {
        if (block_st_index + num_group_per_block * GroupSize <= input_len) {
            if constexpr (Bit==8)
                quant_fp32::shared_to_global<4>(shared, output + (block_st_index / 4), num_group_per_block * GroupSize / 4);
            else
                quant_fp32::shared_to_global<8>(shared, output + (block_st_index / 8), num_group_per_block * GroupSize / 8);
        } else {
            if constexpr (Bit==8)
                quant_fp32::shared_to_global<4>(shared, output + (block_st_index / 4), (input_len - block_st_index) / 4);
            else
                quant_fp32::shared_to_global<8>(shared, output + (block_st_index / 8), (input_len - block_st_index) / 8);
        }
    }

    // save scale and topk
    int64_t scale_index;
    if (isMainThread) {
        if constexpr (Bit == 8) {
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


            *((float*)(output + scale_index)) = topk_ret.scale;
            if constexpr (TopK == 1) {
                output[scale_index + 1] = topk_ret.top1_index;
                ((float*)output)[scale_index + 2] = topk_ret.top1;
            } else if constexpr (TopK == 2) {
                output[scale_index + 1] = (topk_ret.top1_index | (topk_ret.top2_index << 8));
                ((float*)output)[scale_index + 2] = topk_ret.top1;
                ((float*)output)[scale_index + 3] = topk_ret.top2;
            }

        } else if constexpr (Bit == 4) {
            if constexpr (Even)
                scale_index = num_group_per_block * gridDim.x * GroupSize / 8;
            else
                scale_index = input_len / 8;

            if constexpr (TopK == 0)
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup));
            else if constexpr (TopK == 1) 
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 3;
            else if constexpr (TopK == 2) 
                scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 4;


            *((float*)(output + scale_index)) = topk_ret.scale;
            if constexpr (TopK == 1) {
                output[scale_index + 1] = topk_ret.top1_index;
                ((float*)output)[scale_index + 2] = topk_ret.top1;
            } else if constexpr (TopK == 2) {
                output[scale_index + 1] = (topk_ret.top1_index | (topk_ret.top2_index << 8));
                ((float*)output)[scale_index + 2] = topk_ret.top1;
                ((float*)output)[scale_index + 3] = topk_ret.top2;
            }
        }
    }
}


template <int GroupSize, int TopK, bool Stochastic, int ThreadsPerGroup, int Bit, QuantType Type, bool Even>
__global__ void quant_kernel_fp32_compact(uint32_t* input, uint32_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len) {
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 8 == 0) && (Bit == 4)), "GroupSize should be multiple of 4 for 8bit quantization or multiple of 8 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");
    static_assert(GroupSize <= 256, "GroupSize should be less than 256.");
    static_assert(ThreadsPerGroup == 1, "Only Support ThreadsPerGroup == 1.");

    extern __shared__ uint16_t _shared[];
    uint32_t* shared = (uint32_t*)_shared;

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    if constexpr (Even) 
        quant_fp32::global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
    else {
        if (block_st_index + num_group_per_block * GroupSize <= input_len)
            quant_fp32::global_to_shared<1>(input + block_st_index, shared, num_group_per_block * GroupSize);
        else
            quant_fp32::global_to_shared<1>(input + block_st_index, shared, input_len - block_st_index);
    }

    __syncthreads();

    // get max abs value
    quant_fp32::TopKRet  topk_ret;
    if constexpr (Even)
        quant_fp32::get_topk_and_scale<TopK, GroupSize, ThreadsPerGroup>((float*)shared, topk_ret);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            quant_fp32::get_topk_and_scale<TopK, GroupSize, ThreadsPerGroup>((float*)shared, topk_ret);
        else {
            if constexpr (ThreadsPerGroup > 1) {
                __syncthreads(); // get_topk_and_scale has __syncthreads() inside when ThreadsPerGroup > 1
            }   
        }
    }

    // quantization
    if constexpr (Even) 
        quant_fp32::quant_loop<GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, topk_ret.scale, seed);
    else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len)
            quant_fp32::quant_loop<GroupSize, Stochastic, ThreadsPerGroup, Bit, Type>(shared, topk_ret.scale, seed);
    }

    // rearrange shared memory
    if constexpr (Even) {
        int64_t group_st = threadIdx.x * GroupSize;
        int64_t num_element = GroupSize / (Bit == 4 ? 8 : 4);
        for (int i = 1; i < num_element; ++i)
            shared[group_st + i] = shared[group_st + i * (Bit == 4 ? 8 : 4)];

        ((float*)shared)[group_st + num_element] = topk_ret.scale;
        if constexpr (TopK == 1) {
            shared[group_st + num_element + 1] = topk_ret.top1_index;
            ((float*)shared)[group_st + num_element + 2] = topk_ret.top1;
        } else if constexpr (TopK == 2) {
            shared[group_st + num_element + 1] = topk_ret.top1_index | (topk_ret.top2_index << 8);
            ((float*)shared)[group_st + num_element + 2] = topk_ret.top1;
            ((float*)shared)[group_st + num_element + 3] = topk_ret.top2;
        }
    } else {
        if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= input_len) {
            int64_t group_st = threadIdx.x * GroupSize;
            int64_t num_element = GroupSize / (Bit == 4 ? 8 : 4);
            for (int i = 1; i < num_element; ++i)
                shared[group_st + i] = shared[group_st + i * (Bit == 4 ? 8 : 4)];

            ((float*)shared)[group_st + num_element] = topk_ret.scale;
            if constexpr (TopK == 1) {
                shared[group_st + num_element + 1] = topk_ret.top1_index;
                ((float*)shared)[group_st + num_element + 2] = topk_ret.top1;
            } else if constexpr (TopK == 2) {
                shared[group_st + num_element + 1] = topk_ret.top1_index | (topk_ret.top2_index << 8);
                ((float*)shared)[group_st + num_element + 2] = topk_ret.top1;
                ((float*)shared)[group_st + num_element + 3] = topk_ret.top2;
            }
        }
    }


    __syncthreads();

    // save to global memory
    uint32_t buffer[4];
    int64_t element_per_group = GroupSize / (Bit == 4 ? 8 : 4) + 1;
    if constexpr (TopK == 1) element_per_group += 2;
    else if constexpr (TopK == 2) element_per_group += 3;
    int64_t total_element = element_per_group * num_group_per_block;
    int64_t global_st = total_element * blockIdx.x;
    if constexpr (!Even) {
        total_element = min(total_element, ((input_len - num_group_per_block * blockIdx.x * GroupSize) / GroupSize) * element_per_group);
    }
    for (int64_t i = threadIdx.x * 4, group_id, group_offset; i < total_element; i += 4 * blockDim.x) {
        if (i + 8 > total_element) {
            for (int64_t j = 0; j < total_element - i; ++j) {
                group_id = (i + j) / element_per_group;
                group_offset = (i + j) - group_id * element_per_group;
                *(output + global_st + i + j) = *(shared + GroupSize * group_id + group_offset);
            }
        } else {
            for (int64_t j = 0; j < 4; ++j) {
                group_id = (i + j) / element_per_group;
                group_offset = (i + j) - group_id * element_per_group;
                buffer[j] = *(shared + GroupSize * group_id + group_offset);
            }
            *((int4*)(output + global_st + i)) = *((int4*)buffer);
        }
    }
}

