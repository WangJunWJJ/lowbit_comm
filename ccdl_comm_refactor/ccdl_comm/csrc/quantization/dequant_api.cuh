#include <torch/extension.h>
#include <vector>
#include "enum.cuh"

void inplace_dequantize(torch::Tensor input, torch::Tensor output, int64_t group_size, int64_t topk, int64_t bit, ReduceOP reduce_op = ReduceOP::NONE, QuantType quant_type = QuantType::Linear, bool compact=false);
torch::Tensor dequantize(torch::Tensor input, int64_t group_size, int64_t topk, int64_t bit, ReduceOP reduce_op = ReduceOP::NONE, QuantType quant_type = QuantType::Linear, DType dtype = DType::FP16, bool compact=false);
void inplace_dequantize_reduce(std::vector<torch::Tensor> inputs, torch::Tensor output, int64_t group_size, int64_t topk, int64_t bit, QuantType quant_type = QuantType::Linear, bool compact=false);
torch::Tensor dequantize_reduce(std::vector<torch::Tensor> inputs, int64_t group_size, int64_t topk, int64_t bit, QuantType quant_type = QuantType::Linear, DType dtype = DType::FP16, bool compact=false);
