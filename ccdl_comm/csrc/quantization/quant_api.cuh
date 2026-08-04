#include <torch/extension.h>
#include <c10/util/Optional.h>
#include "enum.cuh"

void inplace_quantize(torch::Tensor input, torch::Tensor output, int64_t group_size, int64_t topk, bool stochastic, int64_t bit, QuantType quant_type = QuantType::Linear, bool compact=false);
torch::Tensor quantize(torch::Tensor input, int64_t group_size, int64_t topk, bool stochastic, int64_t bit, QuantType quant_type = QuantType::Linear, bool compact=false);
bool inplace_quantize_pack(
    torch::Tensor input,
    torch::Tensor output,
    c10::optional<torch::Tensor> residual,
    int64_t group_size,
    int64_t topk,
    bool stochastic,
    int64_t bit,
    QuantType quant_type = QuantType::Linear,
    bool compact = true
);
