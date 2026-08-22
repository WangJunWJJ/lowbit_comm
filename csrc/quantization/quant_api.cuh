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
bool inplace_quantize_pack_gradient_error_feedback(
    torch::Tensor input,
    torch::Tensor output,
    c10::optional<torch::Tensor> residual,
    torch::Tensor candidate_residual,
    int64_t group_size
);
bool inplace_quantize_parameter_delta(
    torch::Tensor master,
    torch::Tensor model,
    torch::Tensor output,
    int64_t valid_numel,
    int64_t group_size,
    int64_t topk,
    bool stochastic,
    int64_t bit,
    QuantType quant_type = QuantType::Linear,
    bool compact = true
);
bool try_inplace_quantize_parameter_delta(
    const torch::Tensor& master,
    const torch::Tensor& model,
    torch::Tensor& output,
    int64_t valid_numel,
    int64_t group_size
);
bool try_inplace_shard_quantize_pack(
    const torch::Tensor& input,
    torch::Tensor& packed,
    int64_t logical_shard_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size
);
bool try_inplace_shard_quantize_pack_gradient_error_feedback(
    const torch::Tensor& input,
    torch::Tensor& packed,
    const c10::optional<torch::Tensor>& residual,
    torch::Tensor& candidate_residual,
    int64_t logical_shard_length,
    int64_t transport_shard_length,
    int64_t world_size,
    int64_t group_size
);
