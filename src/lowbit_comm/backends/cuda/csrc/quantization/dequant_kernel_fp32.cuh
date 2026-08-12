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

namespace dequant_fp32 {

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



template <int GroupSize, int ThreadsPerGroup, int Bit, QuantType Type>
__device__ void dequant_loop(uint32_t* srd, float scale) {
    int64_t st = threadIdx.x * GroupSize / ThreadsPerGroup;

    if constexpr (Type == QuantType::Linear) {
        scale = scale / (Bit == 8 ? 127.0 : 7.0);
    }

    uint32_t raw_data = 0;
    for (int64_t i = GroupSize / ThreadsPerGroup - 1, _i; i >= 0; --i) {
        float value;
        if constexpr (Bit == 8) {
            raw_data >>= 8;
            if ((i & 3) == 3)
                raw_data = srd[st + i - 3];
            value = dequant_inner<float, 8, Type>(raw_data & 255, scale);
        } else {
            raw_data >>= 4;
            if ((i & 7) == 7)
                raw_data = srd[st + i - 7];
            value = dequant_inner<float, 4, Type>(raw_data & 15, scale);
        }
        ((float*)srd)[st + i] = value;
    }
} 

template <ReduceOP OP>
__device__ void shared_to_global_op(float* srd, float* glb, int64_t len) {
    int64_t step = blockDim.x;
    for (int64_t index = threadIdx.x; index < len; index += step) {
        if constexpr (OP == ReduceOP::SUM) {
            glb[index] += srd[index];
        } else if constexpr (OP == ReduceOP::MAX) {
            glb[index] = fmax(glb[index], srd[index]);
        } else if constexpr (OP == ReduceOP::MIN) {
            glb[index] = fmin(glb[index], srd[index]);
        } else {
            glb[index] = srd[index];
        }
    }
}

}

template <int GroupSize, int TopK, int ThreadsPerGroup, int Bit, ReduceOP OP, QuantType Type, bool Even>
__global__ void dequant_kernel_fp32(uint32_t* input, uint32_t* output, int64_t length) {
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 8 == 0) && (Bit == 4)), "GroupSize should be multiple of 4 for 8bit quantization or multiple of 8 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");

    extern __shared__ uint16_t _shared[];
    uint32_t* shared = (uint32_t*)_shared;

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    if constexpr (Even) {
        if constexpr (Bit==8) {
            dequant_fp32::global_to_shared<4>(input + block_st_index / 4, shared, num_group_per_block * GroupSize / 4);
        } else {
            dequant_fp32::global_to_shared<8>(input + block_st_index / 8, shared, num_group_per_block * GroupSize / 8);
        }
    } else {
        if (block_st_index + num_group_per_block * GroupSize > length) {
            if constexpr (Bit==8) {
                dequant_fp32::global_to_shared<4>(input + block_st_index / 4, shared, (length - block_st_index) / 4);
            } else {
                dequant_fp32::global_to_shared<8>(input + block_st_index / 8, shared, (length - block_st_index) / 8);
            }
        } else {
            if constexpr (Bit==8) {
                dequant_fp32::global_to_shared<4>(input + block_st_index / 4, shared, num_group_per_block * GroupSize / 4);
            } else {
                dequant_fp32::global_to_shared<8>(input + block_st_index / 8, shared, num_group_per_block * GroupSize / 8);
            }
        }
    }
    
    __syncthreads();

    // load scale 
    int64_t scale_index;
    if constexpr (Even) 
        scale_index = num_group_per_block * gridDim.x * GroupSize / (Bit == 8 ? 4 : 8);
    else
        scale_index = length / (Bit == 8 ? 4 : 8);
    
    if constexpr (TopK == 0)
        scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup));
    else if constexpr (TopK == 1) 
        scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 3;
    else if constexpr (TopK == 2) 
        scale_index += (num_group_per_block * blockIdx.x + (threadIdx.x / ThreadsPerGroup)) * 4;

    if constexpr (Even) {
        float scale = ((float*)input)[scale_index];
        dequant_fp32::dequant_loop<GroupSize, ThreadsPerGroup, Bit, Type>(shared, scale);
    } else {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup <= length) {
            float scale = ((float*)input)[scale_index];
            dequant_fp32::dequant_loop<GroupSize, ThreadsPerGroup, Bit, Type>(shared, scale);
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
            uint32_t top_index = input[scale_index + 1];
            float top_value = ((float*)input)[scale_index + 2];
            assert (top_index < GroupSize);
            ((float*)shared)[group_st_index + top_index] = top_value;
        } else if constexpr (TopK == 2) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            uint32_t top_index = input[scale_index + 1];
            uint32_t top_index1 = top_index & 0x00FF;
            uint32_t top_index2 = top_index >> 8;
            float top_value1 = ((float*)input)[scale_index + 2];
            float top_value2 = ((float*)input)[scale_index + 3];
            assert (top_index1 < GroupSize);
            assert (top_index2 < GroupSize);
            ((float*)shared)[group_st_index + top_index1] = top_value1;
            ((float*)shared)[group_st_index + top_index2] = top_value2;
        }
    }

    __syncthreads();

    // save to output
    if constexpr (Even) {
        dequant_fp32::shared_to_global_op<OP>((float*)shared, (float*)(output + block_st_index), num_group_per_block * GroupSize);
    } else {
        if (block_st_index + num_group_per_block * GroupSize > length)
            dequant_fp32::shared_to_global_op<OP>((float*)shared, (float*)(output + block_st_index), (length - block_st_index));
        else
            dequant_fp32::shared_to_global_op<OP>((float*)shared, (float*)(output + block_st_index), num_group_per_block * GroupSize);
    }
}   

template <int GroupSize, int TopK, int ThreadsPerGroup, int Bit, ReduceOP OP, QuantType Type, bool Even>
__global__ void dequant_kernel_fp32_compact(uint32_t* input, uint32_t* output, int64_t length) {
    static_assert(Bit == 4 || Bit == 8, "Only Support 4bit or 8bit quantization.");
    static_assert(GroupSize % ThreadsPerGroup == 0, "GroupSize should be multiple of ThreadsPerGroup.");
    static_assert((((GroupSize/ThreadsPerGroup) % 4 == 0) && (Bit == 8)) || (((GroupSize/ThreadsPerGroup) % 8 == 0) && (Bit == 4)), "GroupSize should be multiple of 4 for 8bit quantization or multiple of 8 for 4bit quantization.");
    static_assert(TopK <= 2, "Only Support TopK <= 2.");
    static_assert(ThreadsPerGroup == 1, "Only Support ThreadsPerGroup == 1");

    extern __shared__ uint16_t _shared[];
    uint32_t* shared = (uint32_t*)_shared;

    const int64_t num_group_per_block = blockDim.x / ThreadsPerGroup;
    const int64_t block_st_index = blockIdx.x * num_group_per_block * GroupSize;

    // load to shared memory
    {
        u_int32_t buffer[4];
        int64_t element_per_group = GroupSize / (Bit == 4 ? 8 : 4) + 1;
        if constexpr (TopK == 1) element_per_group += 2;
        else if constexpr (TopK == 2) element_per_group += 3;
        int64_t total_element = element_per_group * num_group_per_block;
        int64_t global_st = total_element * blockIdx.x;
        if constexpr (!Even) {
            total_element = min(total_element, ((length - block_st_index) / GroupSize) * element_per_group);
        }
        for (int64_t i = threadIdx.x * 4, group_id, group_offset; i < total_element; i += 4 * blockDim.x) {
            if (i + 4 > total_element) {
                for (int64_t j = 0; j < total_element - i; ++j) {
                    group_id = (i + j) / element_per_group;
                    group_offset = (i + j) - group_id * element_per_group;
                    *(shared + GroupSize * group_id + group_offset) = *(input + global_st + i + j);
                }
            } else {
                *((int4*)buffer) = *((int4 *)(input + global_st + i));
                for (int64_t j = 0; j < 4; ++j) {
                    group_id = (i + j) / element_per_group;
                    group_offset = (i + j) - group_id * element_per_group;
                    *(shared + GroupSize * group_id + group_offset) = buffer[j];
                }
            }
        }
    }


    __syncthreads();

    // rearrange shared memory, load scale and topk
    float scale, top1, top2;
    uint32_t top1_index, top2_index;
    {
        if constexpr (Even) {
            int64_t group_st = threadIdx.x * GroupSize;
            int64_t num_element = GroupSize / (Bit == 4 ? 8 : 4);
            scale = ((float*)shared)[group_st + num_element];
            if constexpr (TopK == 1) {
                top1_index = shared[group_st + num_element + 1];
                top1 = ((float*)shared)[group_st + num_element + 2];
            } else if constexpr (TopK == 2) {
                top1_index = shared[group_st + num_element + 1];
                top1 = ((float*)shared)[group_st + num_element + 2];
                top2 = ((float*)shared)[group_st + num_element + 3];
                top2_index = (top1_index >> 8) & 255;
                top1_index = top1_index & 255;
            }
            for (int i = num_element - 1; i; --i) {
                shared[group_st + i * (Bit == 4 ? 8 : 4)] = shared[group_st + i];
            }
        } else {
            if (block_st_index + (threadIdx.x + 1) * (GroupSize / ThreadsPerGroup) <= length) {
                int64_t group_st = threadIdx.x * GroupSize;
                int64_t num_element = GroupSize / (Bit == 4 ? 8 : 4);
                scale = ((float*)shared)[group_st + num_element];
                if constexpr (TopK == 1) {
                    top1_index = shared[group_st + num_element + 1];
                    top1 = ((float*)shared)[group_st + num_element + 2];
                } else if constexpr (TopK == 2) {
                    top1_index = shared[group_st + num_element + 1];
                    top1 = ((float*)shared)[group_st + num_element + 2];
                    top2 = ((float*)shared)[group_st + num_element + 3];
                    top2_index = (top1_index >> 8) & 255;
                    top1_index = top1_index & 255;
                }
                for (int i = num_element - 1; i; --i) {
                    shared[group_st + i * (Bit == 4 ? 8 : 4)] = shared[group_st + i];
                }
            }
        }
    }

    // dequant
    if constexpr (Even) {
        dequant_fp32::dequant_loop<GroupSize, ThreadsPerGroup, Bit, Type>(shared, scale);
    } else {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup <= length) {
            dequant_fp32::dequant_loop<GroupSize, ThreadsPerGroup, Bit, Type>(shared, scale);
        }
    }

    bool isMainThread = true;
    if constexpr (!Even) {
        if (block_st_index + (threadIdx.x + 1) * GroupSize / ThreadsPerGroup > length)
            isMainThread = false;
    }
    if (isMainThread) {
        if constexpr (TopK == 1) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            assert (top1_index < GroupSize);
            ((float*)shared)[group_st_index + top1_index] = top1;
        } else if constexpr (TopK == 2) {
            int64_t group_st_index = threadIdx.x / ThreadsPerGroup * GroupSize;
            assert (top1_index < GroupSize);
            assert (top2_index < GroupSize);
            ((float*)shared)[group_st_index + top1_index] = top1;
            ((float*)shared)[group_st_index + top2_index] = top2;
        }
    }

    __syncthreads();

    // save to output
    if constexpr (Even) {
        dequant_fp32::shared_to_global_op<OP>((float*)shared, (float*)(output + block_st_index), num_group_per_block * GroupSize);
    } else {
        if (block_st_index + num_group_per_block * GroupSize > length)
            dequant_fp32::shared_to_global_op<OP>((float*)shared, (float*)(output + block_st_index), (length - block_st_index));
        else
            dequant_fp32::shared_to_global_op<OP>((float*)shared, (float*)(output + block_st_index), num_group_per_block * GroupSize);
    }
}   