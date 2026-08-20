#include "fulltensor_plan.h"

#include "../quantization/dequant_api.cuh"
#include "../quantization/enum.cuh"
#include "../quantization/quant_api.cuh"

#include <torch/csrc/distributed/c10d/Types.hpp>

#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace ccdl_comm {

namespace {

constexpr std::array<const char*, 16> kConfigKeys{
    "accumulation_dtype",
    "collective",
    "compression",
    "dtype",
    "gathered_payload_bytes",
    "group_count",
    "group_size",
    "logical_numel",
    "numel",
    "output_bytes",
    "padded_numel",
    "payload_bytes_per_rank",
    "rank",
    "reduction",
    "workspace_bytes",
    "world_size"};

py::handle required(const py::dict& config, const char* key) {
  if (!config.contains(py::str(key))) {
    throw py::value_error(std::string("missing CUDA config field: ") + key);
  }
  return config[py::str(key)];
}

std::string exact_string(const py::dict& config, const char* key) {
  py::handle value = required(config, key);
  if (!PyUnicode_CheckExact(value.ptr())) {
    throw py::value_error(std::string("CUDA config field must be str: ") + key);
  }
  return py::cast<std::string>(value);
}

int64_t exact_nonnegative_int(const py::dict& config, const char* key) {
  py::handle value = required(config, key);
  if (!PyLong_CheckExact(value.ptr())) {
    throw py::value_error(std::string("CUDA config field must be int: ") + key);
  }
  const int64_t result = py::cast<int64_t>(value);
  if (result < 0) {
    throw py::value_error(
        std::string("CUDA config field must be non-negative: ") + key);
  }
  return result;
}

void validate_exact_keys(const py::dict& config) {
  if (!PyDict_CheckExact(config.ptr())) {
    throw py::value_error("CUDA config must be an exact dict");
  }
  if (config.size() != kConfigKeys.size()) {
    throw py::value_error("CUDA config fields are invalid");
  }
  for (const char* key : kConfigKeys) {
    required(config, key);
  }
}

FullTensorCompression parse_compression(const py::dict& config) {
  const std::string value = exact_string(config, "compression");
  if (value == "none") {
    return FullTensorCompression::kNative;
  }
  if (value == "int8") {
    return FullTensorCompression::kInt8;
  }
  throw py::value_error("CUDA FullTensor compression is unsupported");
}

FullTensorReduction parse_reduction(const py::dict& config) {
  const std::string value = exact_string(config, "reduction");
  if (value == "sum") {
    return FullTensorReduction::kSum;
  }
  if (value == "mean") {
    return FullTensorReduction::kMean;
  }
  throw py::value_error("CUDA FullTensor reduction is unsupported");
}

at::ScalarType parse_dtype(const py::dict& config) {
  const std::string value = exact_string(config, "dtype");
  if (value == "fp16") {
    return at::kHalf;
  }
  if (value == "bf16") {
    return at::kBFloat16;
  }
  throw py::value_error("CUDA FullTensor dtype is unsupported");
}

void validate_descriptor(
    const py::dict& config,
    FullTensorCompression compression,
    int64_t numel,
    int64_t world_size,
    int64_t workspace_bytes) {
  if (exact_string(config, "accumulation_dtype") != "fp32") {
    throw py::value_error("CUDA FullTensor accumulation must be fp32");
  }
  const std::string collective = exact_string(config, "collective");
  py::handle group_size_value = required(config, "group_size");
  if (compression == FullTensorCompression::kNative) {
    if (collective != "native" || !group_size_value.is_none()) {
      throw py::value_error("CUDA native descriptor is inconsistent");
    }
    if (workspace_bytes != 0) {
      throw py::value_error("CUDA native workspace must be zero");
    }
  } else {
    if (collective != "compressed_all_gather_reduce" ||
        !PyLong_CheckExact(group_size_value.ptr())) {
      throw py::value_error("CUDA INT8 descriptor is inconsistent");
    }
  }
  if (exact_nonnegative_int(config, "logical_numel") != numel) {
    throw py::value_error("CUDA logical numel mismatch");
  }
  const int64_t padded_numel =
      exact_nonnegative_int(config, "padded_numel");
  const int64_t group_count =
      exact_nonnegative_int(config, "group_count");
  const int64_t payload_bytes =
      exact_nonnegative_int(config, "payload_bytes_per_rank");
  const int64_t gathered_bytes =
      exact_nonnegative_int(config, "gathered_payload_bytes");
  const int64_t output_bytes =
      exact_nonnegative_int(config, "output_bytes");
  if (compression == FullTensorCompression::kNative) {
    if (numel > std::numeric_limits<int64_t>::max() / 2 ||
        padded_numel != numel || group_count != 0 || payload_bytes != 0 ||
        gathered_bytes != 0 || output_bytes != numel * 2) {
      throw py::value_error("CUDA native layout is inconsistent");
    }
  } else {
    const int64_t group_size = py::cast<int64_t>(group_size_value);
    if (group_size != 16 && group_size != 32 && group_size != 64) {
      throw py::value_error("CUDA INT8 group size is unsupported");
    }
    if (numel > std::numeric_limits<int64_t>::max() - group_size + 1) {
      throw py::value_error("CUDA INT8 layout size overflow");
    }
    const int64_t expected_groups =
        (numel + group_size - 1) / group_size;
    if (expected_groups >
        std::numeric_limits<int64_t>::max() / (group_size + 2)) {
      throw py::value_error("CUDA INT8 layout size overflow");
    }
    const int64_t expected_payload =
        expected_groups * (group_size + 2);
    if (expected_payload >
        std::numeric_limits<int64_t>::max() / (world_size + 1)) {
      throw py::value_error("CUDA INT8 layout size overflow");
    }
    if (numel > std::numeric_limits<int64_t>::max() / 2 ||
        padded_numel != expected_groups * group_size ||
        group_count != expected_groups ||
        payload_bytes != expected_payload ||
        gathered_bytes != expected_payload * world_size ||
        output_bytes != numel * 2 ||
        workspace_bytes != expected_payload * (world_size + 1)) {
      throw py::value_error("CUDA INT8 layout is inconsistent");
    }
  }
}

}  // namespace

FullTensorPlan::FullTensorPlan(
    FullTensorCompression compression,
    FullTensorReduction reduction,
    at::ScalarType dtype,
    int64_t numel,
    int64_t rank,
    int64_t world_size,
    int64_t group_size,
    int64_t payload_bytes_per_rank,
    int64_t workspace_bytes,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group)
    : compression_(compression),
      reduction_(reduction),
      dtype_(dtype),
      numel_(numel),
      rank_(rank),
      world_size_(world_size),
      group_size_(group_size),
      payload_bytes_per_rank_(payload_bytes_per_rank),
      workspace_bytes_(workspace_bytes),
      process_group_(std::move(process_group)),
      workspace_pool_(std::make_shared<WorkspacePool>(workspace_bytes)),
      plan_id_(allocate_cuda_plan_id()) {}

void FullTensorPlan::validate_input(const torch::Tensor& input) const {
  if (!input.defined() || !input.is_cuda()) {
    throw CudaExecutionError("FullTensor input must be a CUDA tensor");
  }
  if (!input.is_contiguous()) {
    throw CudaExecutionError("FullTensor input must be contiguous");
  }
  if (input.scalar_type() != dtype_) {
    throw CudaExecutionError("FullTensor input dtype mismatch");
  }
  if (input.numel() != numel_) {
    throw CudaExecutionError("FullTensor input numel mismatch");
  }
}

std::shared_ptr<CudaWork> FullTensorPlan::execute_native(
    torch::Tensor input) {
  std::vector<at::Tensor> tensors{input};
  c10d::AllreduceOptions options;
  options.reduceOp = c10d::ReduceOp::SUM;
  auto transport = process_group_->allreduce(tensors, options);
  if (!transport) {
    throw CudaExecutionError("NCCL all-reduce returned no Work");
  }
  bool completed = false;
  {
    py::gil_scoped_release release;
    completed = transport->wait();
  }
  if (!completed) {
    throw CudaExecutionError("NCCL all-reduce did not complete");
  }
  if (reduction_ == FullTensorReduction::kMean) {
    input.div_(world_size_);
  }
  const LaunchToken token{plan_id_, allocate_cuda_sequence(next_sequence_)};
  return std::make_shared<CudaWork>(py::cast(input), token, nullptr);
}

std::shared_ptr<CudaWork> FullTensorPlan::execute_int8(
    torch::Tensor input) {
  if (numel_ == 0) {
    const LaunchToken token{
        plan_id_, allocate_cuda_sequence(next_sequence_)};
    return std::make_shared<CudaWork>(py::cast(input), token, nullptr);
  }

  std::unique_ptr<WorkspaceLease> lease =
      workspace_pool_->acquire(workspace_bytes_);
  const torch::Tensor& storage = lease->storage();
  torch::Tensor send = storage.narrow(0, 0, payload_bytes_per_rank_);
  torch::Tensor gathered = storage.narrow(
      0,
      payload_bytes_per_rank_,
      payload_bytes_per_rank_ * world_size_);
  if (!inplace_quantize_pack(
          input,
          send,
          c10::nullopt,
          group_size_,
          0,
          false,
          8,
          QuantType::Linear,
          true)) {
    throw CudaExecutionError("INT8 compact quantize-pack is unsupported");
  }

  std::vector<at::Tensor> receive_views;
  receive_views.reserve(world_size_);
  for (int64_t rank = 0; rank < world_size_; ++rank) {
    receive_views.push_back(gathered.narrow(
        0,
        rank * payload_bytes_per_rank_,
        payload_bytes_per_rank_));
  }
  std::vector<std::vector<at::Tensor>> outputs{receive_views};
  std::vector<at::Tensor> inputs{send};
  c10d::AllgatherOptions options;
  auto transport = process_group_->allgather(outputs, inputs, options);
  if (!transport) {
    throw CudaExecutionError("NCCL all-gather returned no Work");
  }
  bool completed = false;
  {
    py::gil_scoped_release release;
    completed = transport->wait();
  }
  if (!completed) {
    throw CudaExecutionError("NCCL all-gather did not complete");
  }

  const float inverse_divisor =
      reduction_ == FullTensorReduction::kMean
      ? 1.0f / static_cast<float>(world_size_)
      : 1.0f;
  if (!try_inplace_dequantize_reduce_fused(
          receive_views,
          input,
          group_size_,
          0,
          8,
          QuantType::Linear,
          true,
          inverse_divisor)) {
    throw CudaExecutionError("INT8 fused dequant-reduce is unsupported");
  }
  const LaunchToken token{plan_id_, allocate_cuda_sequence(next_sequence_)};
  return std::make_shared<CudaWork>(
      py::cast(input), token, std::move(lease));
}

std::shared_ptr<CudaWork> FullTensorPlan::execute(torch::Tensor input) {
  try {
    validate_input(input);
    if (compression_ == FullTensorCompression::kNative) {
      return execute_native(std::move(input));
    }
    return execute_int8(std::move(input));
  } catch (const CudaExecutionError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaExecutionError(
        std::string("FullTensor execution failed: ") + error.what());
  } catch (...) {
    throw CudaExecutionError("FullTensor execution failed");
  }
}

std::shared_ptr<FullTensorPlan> create_fulltensor_plan(
    py::dict config,
    py::object process_group) {
  validate_exact_keys(config);
  const FullTensorCompression compression = parse_compression(config);
  const FullTensorReduction reduction = parse_reduction(config);
  const at::ScalarType dtype = parse_dtype(config);
  const int64_t numel = exact_nonnegative_int(config, "numel");
  const int64_t rank = exact_nonnegative_int(config, "rank");
  const int64_t world_size = exact_nonnegative_int(config, "world_size");
  const int64_t workspace_bytes =
      exact_nonnegative_int(config, "workspace_bytes");
  if (world_size != 2 && world_size != 4) {
    throw py::value_error("CUDA ProcessGroup world size is unsupported");
  }
  if (rank >= world_size) {
    throw py::value_error("CUDA ProcessGroup rank is invalid");
  }
  validate_descriptor(
      config, compression, numel, world_size, workspace_bytes);
  int64_t group_size = 0;
  if (compression == FullTensorCompression::kInt8) {
    group_size = py::cast<int64_t>(required(config, "group_size"));
  }
  const int64_t payload_bytes_per_rank =
      exact_nonnegative_int(config, "payload_bytes_per_rank");

  c10::intrusive_ptr<c10d::ProcessGroup> group;
  try {
    group = process_group.cast<c10::intrusive_ptr<c10d::ProcessGroup>>();
  } catch (const py::cast_error& error) {
    throw py::value_error(
        std::string("CUDA backend requires a c10d ProcessGroup: ") +
        error.what());
  }
  if (!group) {
    throw py::value_error("CUDA backend requires a c10d ProcessGroup");
  }
  if (group->getBackendName() != "nccl") {
    throw py::value_error("CUDA FullTensor execution requires NCCL");
  }
  if (group->getRank() != rank) {
    throw py::value_error("CUDA ProcessGroup rank mismatch");
  }
  if (group->getSize() != world_size) {
    throw py::value_error("CUDA ProcessGroup world size mismatch");
  }
  return std::make_shared<FullTensorPlan>(
      compression,
      reduction,
      dtype,
      numel,
      rank,
      world_size,
      group_size,
      payload_bytes_per_rank,
      workspace_bytes,
      std::move(group));
}

void bind_fulltensor_plan(py::module_& module) {
  py::class_<FullTensorPlan, std::shared_ptr<FullTensorPlan>>(
      module, "FullTensorPlan")
      .def("execute", &FullTensorPlan::execute, py::arg("input"));
  module.def(
      "create_fulltensor_plan",
      &create_fulltensor_plan,
      py::arg("config"),
      py::arg("process_group"));
}

}  // namespace ccdl_comm
