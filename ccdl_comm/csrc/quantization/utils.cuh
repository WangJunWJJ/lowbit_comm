#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define fmax(a, b) (((a) > (b)) ? (a) : (b))
#define fmin(a, b) (((a) > (b)) ? (b) : (a))
#define hfmax(a, b) ((__hgt(a, b)) ? (a) : (b))
#define hfmin(a, b) ((__hgt(a, b)) ? (b) : (a))
#define _clamp(v, a, b) (fmax((fmin(v, b)), a))

#define CLR2_MASK 0xfffffffffffffffe
#define CLR4_MASK 0xfffffffffffffffc
#define CLR2(x) ((x) & CLR2_MASK)
#define CLR4(x) ((x) & CLR4_MASK)

template <int64_t Step>
__device__ void global_to_shared(uint16_t* glb, uint16_t* srd, int64_t len) {
    uint16_t buffer[8];
    int64_t load_per_thread = (len & (-1024)) / 128;
    int64_t st = load_per_thread * threadIdx.x;
    for (int64_t index = 0; index < load_per_thread; index += 8) {
        *((int4*)buffer) = *((int4*)(glb + st + index));
        #pragma unroll
        for (int64_t j = 0; j < 8; ++j)
            srd[(st + index + j) * Step] = buffer[j];
    }
    if (threadIdx.x < 32) { // the first warp
        int64_t remain = len - (len & (-1024));
        if (remain > 0) {
            for (int64_t index = threadIdx.x * 8 + (len & (-1024)); index < len; index += 256) {
                if (index + 8 <= len) {
                    *((int4*)buffer) = *((int4*)(glb + index));
                    #pragma unroll
                    for (int64_t j = 0; j < 8; ++j)
                        srd[(index + j) * Step] = buffer[j];
                } else {
                    for (int64_t j = 0; j < 8; ++j) {
                        if (index + j < len)
                            srd[(index + j) * Step] = glb[(index + j) * Step];
                    }
                }
            }
        }
    }
}

template <int64_t Step>
__device__ void shared_to_global(uint16_t* srd, uint16_t* glb, int64_t len) {
    uint16_t buffer[8];
    int64_t save_per_thread = (len & (-1024)) / 128;
    int64_t st = save_per_thread * threadIdx.x;
    for (int64_t index = 0; index < save_per_thread; index += 8) {
        #pragma unroll
        for (int64_t j = 0; j < 8; ++j)
            buffer[j] = srd[(st + index + j) * Step];
        *((int4*)(glb + st + index)) = *((int4*)buffer);
    }
    if (threadIdx.x < 32) {
        int64_t remain = len - (len & (-1024));
        if (remain > 0) {
            for (int64_t index = threadIdx.x * 8 + (len & (-1024)); index < len; index += 256) {
                if (index + 8 <= len) {
                    #pragma unroll
                    for (int64_t j = 0; j < 8; ++j)
                        buffer[j] = srd[(index + j) * Step];
                    *((int4*)(glb + index)) = *((int4*)buffer);
                } else {
                    for (int64_t j = 0; j < 8; ++j) {
                        if (index + j < len)
                            glb[index + j] = srd[(index + j) * Step];
                    }
                }
            }
        }
    }
}

template <typename scalar_t> 
__device__ scalar_t float2half(float value) {
    if constexpr (std::is_same<scalar_t, __half>::value)
        return __float2half(value);
    else if constexpr (std::is_same<scalar_t, __nv_bfloat16>::value)
        return __float2bfloat16(value);
    else
        return value;
}

template <typename scalar_t>
__device__ float half2float(scalar_t value) {
    if constexpr (std::is_same<scalar_t, __half>::value)
        return __half2float(value);
    else if constexpr (std::is_same<scalar_t, __nv_bfloat16>::value)
        return __bfloat162float(value);
    else
        return value;
}


template <typename scalar_t>
__device__ int clamp_and_round(scalar_t value, int min_value, int max_value) {
    if constexpr (std::is_same<scalar_t, __half>::value)
        return _clamp(__half2int_rn(value), min_value, max_value);
    else if constexpr (std::is_same<scalar_t, __nv_bfloat16>::value)
        return _clamp(__bfloat162int_rn(value), min_value, max_value);
    else
        return _clamp(__float2int_rn(value), min_value, max_value);

}

cudaStream_t get_current_cuda_stream();