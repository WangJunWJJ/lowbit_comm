#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "enum.cuh"

__constant__ float NormalQmap4[16] = {
    -1.0f, -0.815410147f, -0.655160150, -0.510228027, 
    -0.375324982f, -0.246951370f, -0.122538030, 0, 
    0.122538030, 0.246951370f, 0.375324982f, 0.510228027f, 
    0.655160150, 0.815410147f, 1.0f, INFINITY
};

__constant__ float UniformQmap4[16] = {
    -1.0f, -0.85714286f, -0.71428571f, -0.57142857f, -0.42857143f,
    -0.28571429f, -0.14285714f,  0,  0.14285714f,  0.28571429f,
    0.42857143f,  0.57142857f,  0.71428571f,  0.85714286f,  1.0f,
    INFINITY
};

__constant__ float E3M0Qmap4[16] = {
    -1.0f, -0.5f, -0.25f, -0.125f, -0.0625f, 
    -0.03125f, -0.015625f, 0, 0.015625f, 0.03125f, 
    0.0625f, 0.125f, 0.25f, 0.5f, 1.0f,
    INFINITY
};

__constant__ float E2M1Qmap4[16] = {
    -1.0f, -0.6666666666666666f, -0.5f, -0.3333333333333333f, -0.25f, 
    -0.16666666666666666f, -0.125f, 0, 0.125f, 0.16666666666666666f, 
    0.25f, 0.3333333333333333f, 0.5f, 0.6666666666666666f, 1.0f,
    INFINITY
};

__constant__ float int4_to_float[16] = {
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
};



template <typename scalar_t, bool Stochastic, int Bit> 
__forceinline__ __device__ u_int16_t linear_quant(scalar_t value, float scale, curandStatePhilox4_32_10_t &state) {
    float tmp;
    if constexpr (std::is_same<scalar_t, float>::value)
        tmp = value * scale;
    else
        tmp = half2float<scalar_t>(value) * scale;

    if constexpr (Stochastic) {
        float noise = curand_uniform(&state) - 0.5;
        tmp += noise;
    }
    if constexpr (Bit == 8) {
        return (u_int16_t)(((u_int32_t)clamp_and_round<float>(tmp, -127, 127)) & 255);
    } else if constexpr (Bit == 4) {
        return (u_int16_t)(clamp_and_round<float>(tmp, -7, 7) & 15);
    } else {
        // Not Implemented
        static_assert(Bit == 4 || Bit == 8, "Not Implemented");
    }
}

template <typename scalar_t, int Bit>
__forceinline__ __device__ float linear_dequant(u_int16_t value, float scale) {
    if constexpr (Bit == 4) 
        return int4_to_float[value] * scale;
    else if constexpr (Bit == 8) 
        return static_cast<float>((int8_t)value) * scale;
}

template <typename scalar_t, bool Stochastic, int Bit, const float* Qmap>
__forceinline__ __device__ u_int16_t qmap_quant(scalar_t value, float scale, curandStatePhilox4_32_10_t &state) {
    float tmp = half2float<scalar_t>(value) / scale;
    u_int16_t l = 0, r; // binary search [0, Len-1]
    if constexpr (Bit == 4) {
        r = 14;
    } else if constexpr (Bit == 8) {
        r = 254;
    }
    while (l < r) {
        int mid = (l + r + 1) / 2;
        if (tmp < Qmap[mid]) r = mid - 1;
        else l = mid;
    }
    // Qmap[l] <= tmp || (tmp < Qmap[0] && l == 0)
    if constexpr (Stochastic) {
        float q = curand_uniform(&state) ;
        if (q < (tmp - Qmap[l]) / (Qmap[l+1] - Qmap[l])) l++;
    } else {
        if (tmp - Qmap[l] > (Qmap[l+1] - Qmap[l]) / 2) l++;
    }
    return l;
}

template <typename scalar_t, bool Stochastic, int Bit, QuantType Type>
__forceinline__ __device__ u_int16_t quant_inner(scalar_t value, float scale, curandStatePhilox4_32_10_t &state) {
    if constexpr (Type == QuantType::Linear) {
        return linear_quant<scalar_t, Stochastic, Bit>(value, scale, state);
    } else if constexpr (Type == QuantType::Normal) {
        static_assert(Bit == 4, "Normal Quantization only support 4-bit");
        return qmap_quant<scalar_t, Stochastic, Bit, NormalQmap4>(value, scale, state);
    } else if constexpr (Type == QuantType::Uniform) {
        static_assert(Bit == 4, "Uniform Quantization only support 4-bit");
        return qmap_quant<scalar_t, Stochastic, Bit, UniformQmap4>(value, scale, state);
    } else if constexpr (Type == QuantType::E3M0) {
        static_assert(Bit == 4, "E3M0 Quantization only support 4-bit");
        return qmap_quant<scalar_t, Stochastic, Bit, E3M0Qmap4>(value, scale, state);
    } else if constexpr (Type == QuantType::E2M1) {
        static_assert(Bit == 4, "E2M1 Quantization only support 4-bit");
        return qmap_quant<scalar_t, Stochastic, Bit, E2M1Qmap4>(value, scale, state);
    } else {
        // Not Implemented
        static_assert(Type == QuantType::Linear || Type == QuantType::Normal || Type == QuantType::Uniform, "Not Implemented");
    }
}

template <typename scalar_t, int Bit, QuantType Type>
__forceinline__ __device__ float dequant_inner(u_int16_t value, float scale) {
    if constexpr (Type == QuantType::Linear) {
        return linear_dequant<scalar_t, Bit>(value, scale);
    } else if constexpr (Type == QuantType::Normal) {
        static_assert(Bit == 4, "Normal Quantization only support 4-bit");
        return NormalQmap4[value] * scale;
    } else if constexpr (Type == QuantType::Uniform) {
        static_assert(Bit == 4, "Uniform Quantization only support 4-bit");
        return UniformQmap4[value] * scale;
    } else if constexpr (Type == QuantType::E3M0) {
        static_assert(Bit == 4, "E3M0 Quantization only support 4-bit");
        return E3M0Qmap4[value] * scale;
    } else if constexpr (Type == QuantType::E2M1) {
        static_assert(Bit == 4, "E2M1 Quantization only support 4-bit");
        return E2M1Qmap4[value] * scale;
    } else {
        // Not Implemented
        static_assert(Type == QuantType::Linear || Type == QuantType::Normal || Type == QuantType::Uniform, "Not Implemented");
    }
}