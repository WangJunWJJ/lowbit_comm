#pragma once

#include <torch/extension.h>

void inplace_decode_dynamic_metadata(
    torch::Tensor metadata,
    torch::Tensor descriptors,
    int64_t world_size,
    int64_t dtype_code,
    int64_t bit,
    int64_t group_size,
    int64_t quant_type_code,
    bool compact,
    int64_t layout_generation,
    int64_t max_numel,
    int64_t payload_stride
);
