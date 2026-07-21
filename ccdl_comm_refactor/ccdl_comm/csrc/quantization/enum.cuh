#pragma once

enum class ReduceOP {
    SUM,
    MAX,
    MIN,
    NONE
};

enum class QuantType {
    Linear,
    Normal,
    Uniform,
    E3M0,
    E2M1
};

enum class DType {
    FP16,
    BF16,
    FP32
};