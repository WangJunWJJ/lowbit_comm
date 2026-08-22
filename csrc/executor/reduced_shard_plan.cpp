#include "reduced_shard_plan.h"

#include "../quantization/dequant_api.cuh"
#include "../quantization/quant_api.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ccdl_comm {

namespace {

constexpr std::array<const char*, 21> kConfigKeys{
    "accumulation_dtype",
    "collective",
    "compression",
    "dtype",
    "global_numel",
    "group_size",
    "groups_per_shard",
    "logical_shard_length",
    "numel",
    "offset",
    "output_bytes",
    "output_numel",
    "payload_bytes_per_destination",
    "rank",
    "receive_payload_bytes",
    "reduction",
    "send_payload_bytes",
    "transport_shard_length",
    "valid_length",
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
  const long long converted = PyLong_AsLongLong(value.ptr());
  if (converted == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    throw py::value_error(
        std::string("CUDA config field exceeds signed 64-bit range: ") +
        key);
  }
  const int64_t result = static_cast<int64_t>(converted);
  if (result < 0) {
    throw py::value_error(
        std::string("CUDA config field must be non-negative: ") + key);
  }
  return result;
}

int64_t checked_add(int64_t left, int64_t right, const char* label) {
  if (right > std::numeric_limits<int64_t>::max() - left) {
    throw py::value_error(
        std::string("CUDA ReducedShard ") + label + " size overflow");
  }
  return left + right;
}

int64_t checked_mul(int64_t left, int64_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<int64_t>::max() / left) {
    throw py::value_error(
        std::string("CUDA ReducedShard ") + label + " size overflow");
  }
  return left * right;
}

void validate_exact_keys(const py::dict& config) {
  if (!PyDict_CheckExact(config.ptr())) {
    throw py::value_error("CUDA config must be an exact dict");
  }
  for (const char* key : kConfigKeys) {
    required(config, key);
  }
  const bool has_gradient_error_feedback =
      config.contains(py::str("gradient_error_feedback"));
  const size_t expected_size =
      kConfigKeys.size() + (has_gradient_error_feedback ? 1 : 0);
  if (config.size() != expected_size) {
    throw py::value_error("CUDA config fields are invalid");
  }
}

bool parse_gradient_error_feedback(const py::dict& config) {
  if (!config.contains(py::str("gradient_error_feedback"))) {
    return false;
  }
  py::handle value = required(config, "gradient_error_feedback");
  if (value.ptr() != Py_True && value.ptr() != Py_False) {
    throw py::value_error(
        "CUDA config field must be bool: gradient_error_feedback");
  }
  return value.ptr() == Py_True;
}

ReducedShardCompression parse_compression(const py::dict& config) {
  const std::string value = exact_string(config, "compression");
  if (value == "none") {
    return ReducedShardCompression::kNative;
  }
  if (value == "int8") {
    return ReducedShardCompression::kInt8;
  }
  throw py::value_error("CUDA ReducedShard compression is unsupported");
}

ReducedShardReduction parse_reduction(const py::dict& config) {
  const std::string value = exact_string(config, "reduction");
  if (value == "sum") {
    return ReducedShardReduction::kSum;
  }
  if (value == "mean") {
    return ReducedShardReduction::kMean;
  }
  throw py::value_error("CUDA ReducedShard reduction is unsupported");
}

at::ScalarType parse_dtype(const py::dict& config) {
  const std::string value = exact_string(config, "dtype");
  if (value == "fp16") {
    return at::kHalf;
  }
  if (value == "bf16") {
    return at::kBFloat16;
  }
  throw py::value_error("CUDA ReducedShard dtype is unsupported");
}

void validate_common_layout(
    const py::dict& config,
    int64_t numel,
    int64_t rank,
    int64_t world_size,
    int64_t logical_shard_length) {
  if (exact_string(config, "accumulation_dtype") != "fp32") {
    throw py::value_error("CUDA ReducedShard accumulation must be fp32");
  }
  if (exact_nonnegative_int(config, "global_numel") != numel) {
    throw py::value_error("CUDA ReducedShard global numel mismatch");
  }
  if (exact_nonnegative_int(config, "logical_shard_length") !=
      logical_shard_length) {
    throw py::value_error("CUDA ReducedShard native layout is inconsistent");
  }
  const int64_t expected_offset = std::min(
      checked_mul(rank, logical_shard_length, "shard offset"), numel);
  const int64_t expected_valid_length =
      std::min(logical_shard_length, numel - expected_offset);
  if (exact_nonnegative_int(config, "offset") != expected_offset ||
      exact_nonnegative_int(config, "valid_length") !=
          expected_valid_length) {
    throw py::value_error("CUDA ReducedShard ownership is inconsistent");
  }
  const int64_t output_numel =
      exact_nonnegative_int(config, "output_numel");
  const int64_t output_bytes =
      exact_nonnegative_int(config, "output_bytes");
  if (output_numel != logical_shard_length ||
      output_bytes != checked_mul(logical_shard_length, 2, "output")) {
    throw py::value_error("CUDA ReducedShard output is inconsistent");
  }
}

void validate_native_layout(
    const py::dict& config,
    int64_t numel,
    int64_t world_size,
    int64_t logical_shard_length) {
  if (exact_string(config, "collective") != "native" ||
      !required(config, "group_size").is_none()) {
    throw py::value_error("CUDA ReducedShard native descriptor is inconsistent");
  }
  if (exact_nonnegative_int(config, "transport_shard_length") !=
          logical_shard_length ||
      exact_nonnegative_int(config, "groups_per_shard") != 0 ||
      exact_nonnegative_int(config, "payload_bytes_per_destination") != 0 ||
      exact_nonnegative_int(config, "send_payload_bytes") != 0 ||
      exact_nonnegative_int(config, "receive_payload_bytes") != 0) {
    throw py::value_error("CUDA ReducedShard native layout is inconsistent");
  }
  const int64_t padded_input_numel = checked_mul(
      logical_shard_length, world_size, "padded input");
  const int64_t expected_workspace =
      padded_input_numel == numel
      ? 0
      : checked_mul(padded_input_numel, 2, "padded input");
  if (exact_nonnegative_int(config, "workspace_bytes") !=
      expected_workspace) {
    throw py::value_error("CUDA ReducedShard workspace is inconsistent");
  }
}

void validate_int8_layout(
    const py::dict& config,
    int64_t world_size,
    int64_t logical_shard_length) {
  py::handle group_size_value = required(config, "group_size");
  if (exact_string(config, "collective") !=
          "compressed_reduce_scatter" ||
      !PyLong_CheckExact(group_size_value.ptr())) {
    throw py::value_error(
        "CUDA ReducedShard INT8 descriptor group_size is inconsistent");
  }
  const int64_t group_size = exact_nonnegative_int(config, "group_size");
  if (group_size != 16 && group_size != 32 && group_size != 64) {
    throw py::value_error("CUDA ReducedShard INT8 group_size is unsupported");
  }
  const int64_t numerator =
      checked_add(logical_shard_length, group_size - 1, "transport shard");
  const int64_t groups_per_shard = numerator / group_size;
  const int64_t transport_shard_length =
      checked_mul(groups_per_shard, group_size, "transport shard");
  const int64_t bytes_per_group =
      checked_add(group_size, 2, "payload");
  const int64_t payload_per_destination =
      checked_mul(groups_per_shard, bytes_per_group, "payload");
  const int64_t send_payload =
      checked_mul(payload_per_destination, world_size, "send payload");
  const int64_t receive_payload =
      checked_mul(payload_per_destination, world_size, "receive payload");
  if (exact_nonnegative_int(config, "groups_per_shard") !=
          groups_per_shard ||
      exact_nonnegative_int(config, "transport_shard_length") !=
          transport_shard_length ||
      exact_nonnegative_int(config, "payload_bytes_per_destination") !=
          payload_per_destination ||
      exact_nonnegative_int(config, "send_payload_bytes") != send_payload ||
      exact_nonnegative_int(config, "receive_payload_bytes") !=
          receive_payload ||
      exact_nonnegative_int(config, "workspace_bytes") !=
          checked_add(send_payload, receive_payload, "workspace")) {
    throw py::value_error("CUDA ReducedShard INT8 layout is inconsistent");
  }
}

ReducedShardFailureInjection parse_failure_injection_for_test(
    const std::string& failure) {
  if (failure == "none") {
    return ReducedShardFailureInjection::kNone;
  }
  if (failure == "quant_helper") {
    return ReducedShardFailureInjection::kQuantHelper;
  }
  if (failure == "transport_wait") {
    return ReducedShardFailureInjection::kTransportWait;
  }
  if (failure == "dequant_helper") {
    return ReducedShardFailureInjection::kDequantHelper;
  }
  if (failure == "event_record") {
    return ReducedShardFailureInjection::kEventRecord;
  }
  if (failure == "event_synchronize") {
    return ReducedShardFailureInjection::kEventSynchronize;
  }
  throw py::value_error("ReducedShard failure injection is invalid");
}

}  // namespace

ReducedShardPlan::ReducedShardPlan(
    ReducedShardCompression compression,
    ReducedShardReduction reduction,
    at::ScalarType dtype,
    int64_t numel,
    int64_t logical_shard_length,
    int64_t valid_length,
    int64_t rank,
    int64_t world_size,
    int64_t group_size,
    int64_t transport_shard_length,
    int64_t payload_bytes_per_destination,
    int64_t send_payload_bytes,
    int64_t receive_payload_bytes,
    int64_t workspace_bytes,
    bool gradient_error_feedback,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group)
    : compression_(compression),
      reduction_(reduction),
      dtype_(dtype),
      numel_(numel),
      logical_shard_length_(logical_shard_length),
      valid_length_(valid_length),
      rank_(rank),
      world_size_(world_size),
      group_size_(group_size),
      transport_shard_length_(transport_shard_length),
      payload_bytes_per_destination_(payload_bytes_per_destination),
      send_payload_bytes_(send_payload_bytes),
      receive_payload_bytes_(receive_payload_bytes),
      workspace_bytes_(workspace_bytes),
      gradient_error_feedback_(gradient_error_feedback),
      process_group_(std::move(process_group)),
      workspace_pool_(std::make_shared<WorkspacePool>(workspace_bytes)),
      plan_id_(allocate_cuda_plan_id()) {}

void ReducedShardPlan::validate_input(
    const torch::Tensor& input,
    const c10::optional<torch::Tensor>& committed_residual) const {
  if (!input.defined() || !input.is_cuda()) {
    throw CudaExecutionError("ReducedShard input must be a CUDA tensor");
  }
  if (!input.is_contiguous()) {
    throw CudaExecutionError("ReducedShard input must be contiguous");
  }
  if (input.scalar_type() != dtype_) {
    throw CudaExecutionError("ReducedShard input dtype mismatch");
  }
  if (input.numel() != numel_) {
    throw CudaExecutionError("ReducedShard input numel mismatch");
  }
  if (!gradient_error_feedback_ && committed_residual.has_value()) {
    throw CudaExecutionError(
        "ReducedShard gradient error feedback is disabled");
  }
  if (committed_residual.has_value()) {
    const torch::Tensor& residual = *committed_residual;
    if (!residual.defined() || !residual.is_cuda() ||
        !residual.is_contiguous() ||
        residual.scalar_type() != dtype_ ||
        residual.numel() != numel_ ||
        residual.device() != input.device()) {
      throw CudaExecutionError(
          "ReducedShard committed gradient residual is invalid");
    }
  }
}

std::shared_ptr<CudaWork> ReducedShardPlan::execute_native(
    torch::Tensor input,
    LaunchToken token) {
  side_effects_.mark_allocation();
  torch::Tensor output = torch::empty(
      {logical_shard_length_}, input.options());
  if (numel_ == 0) {
    side_effects_.mark_work_publish();
    return std::make_shared<CudaWork>(py::cast(output), token);
  }

  const int64_t padded_input_numel =
      logical_shard_length_ * world_size_;
  torch::Tensor transport_input = input.view({numel_});
  torch::Tensor padded_input;
  if (padded_input_numel != numel_) {
    side_effects_.mark_allocation();
    padded_input = torch::zeros({padded_input_numel}, input.options());
    padded_input.narrow(0, 0, numel_).copy_(transport_input);
    transport_input = padded_input;
  }
  c10d::ReduceScatterOptions options;
  options.reduceOp = c10d::ReduceOp::SUM;
  side_effects_.mark_transport_launch();
  c10::intrusive_ptr<c10d::Work> transport =
      process_group_->_reduce_scatter_base(output, transport_input, options);
  if (!transport) {
    throw CudaExecutionError("NCCL reduce-scatter returned no Work");
  }
  bool completed = false;
  {
    py::gil_scoped_release release;
    completed = transport->wait();
  }
  if (!completed) {
    throw CudaExecutionError("NCCL reduce-scatter did not complete");
  }
  if (reduction_ == ReducedShardReduction::kMean) {
    side_effects_.mark_kernel_launch();
    output.div_(world_size_);
  }
  side_effects_.mark_work_publish();
  return std::make_shared<CudaWork>(py::cast(output), token);
}

std::shared_ptr<CudaWork> ReducedShardPlan::execute_int8(
    torch::Tensor input,
    LaunchToken token,
    const c10::optional<torch::Tensor>& committed_residual,
    torch::Tensor* candidate_residual) {
  c10::cuda::CUDAGuard device_guard(input.device());
  side_effects_.mark_allocation();
  torch::Tensor output = torch::empty(
      {logical_shard_length_}, input.options());
  if (gradient_error_feedback_) {
    side_effects_.mark_allocation();
    *candidate_residual = torch::empty_like(input);
  }
  if (numel_ == 0) {
    side_effects_.mark_work_publish();
    return std::make_shared<CudaWork>(py::cast(output), token);
  }

  std::unique_ptr<WorkspaceLease> lease =
      [&]() {
        side_effects_.mark_workspace_acquire();
        return workspace_pool_->acquire(workspace_bytes_);
      }();
  const torch::Tensor& storage = lease->storage();
  torch::Tensor send = storage.narrow(0, 0, send_payload_bytes_);
  torch::Tensor receive = storage.narrow(
      0, send_payload_bytes_, receive_payload_bytes_);
  c10::intrusive_ptr<c10d::Work> transport;
  bool workspace_access_started = false;
  try {
    side_effects_.mark_kernel_launch();
    workspace_access_started = true;
    const bool quantized = gradient_error_feedback_
        ? try_inplace_shard_quantize_pack_gradient_error_feedback(
              input,
              send,
              committed_residual,
              *candidate_residual,
              logical_shard_length_,
              transport_shard_length_,
              world_size_,
              group_size_)
        : try_inplace_shard_quantize_pack(
              input,
              send,
              logical_shard_length_,
              transport_shard_length_,
              world_size_,
              group_size_);
    if (!quantized) {
      throw CudaExecutionError(
          "INT8 shard quantize-pack is unsupported");
    }
    if (failure_for_test_ ==
        ReducedShardFailureInjection::kQuantHelper) {
      throw CudaExecutionError("test injected quant_helper failure");
    }

    std::vector<int64_t> splits(
        world_size_, payload_bytes_per_destination_);
    side_effects_.mark_transport_launch();
    transport = process_group_->alltoall_base(
        receive,
        send,
        splits,
        splits,
        c10d::AllToAllOptions{});
    if (!transport) {
      throw CudaExecutionError("NCCL all-to-all returned no Work");
    }
    if (failure_for_test_ ==
        ReducedShardFailureInjection::kTransportWait) {
      throw CudaExecutionError("test injected transport_wait failure");
    }
    bool completed = false;
    {
      py::gil_scoped_release release;
      completed = transport->wait();
    }
    if (!completed) {
      throw CudaExecutionError("NCCL all-to-all did not complete");
    }

    const float inverse_divisor =
        reduction_ == ReducedShardReduction::kMean
        ? 1.0f / static_cast<float>(world_size_)
        : 1.0f;
    side_effects_.mark_kernel_launch();
    if (!try_inplace_shard_dequantize_reduce(
            receive,
            output,
            logical_shard_length_,
            valid_length_,
            transport_shard_length_,
            world_size_,
            group_size_,
            inverse_divisor)) {
      throw CudaExecutionError(
          "INT8 shard fused dequant-reduce is unsupported");
    }
    if (test_delay_armed_.exchange(false, std::memory_order_acq_rel)) {
      launch_shard_dequant_delay_for_test(output);
      test_delay_launch_count_.fetch_add(1, std::memory_order_relaxed);
    }
    if (failure_for_test_ ==
        ReducedShardFailureInjection::kDequantHelper) {
      throw CudaExecutionError("test injected dequant_helper failure");
    }
    CudaEventFailureInjection event_failure =
        CudaEventFailureInjection::kNone;
    if (failure_for_test_ ==
        ReducedShardFailureInjection::kEventRecord) {
      event_failure = CudaEventFailureInjection::kRecord;
    } else if (failure_for_test_ ==
               ReducedShardFailureInjection::kEventSynchronize) {
      event_failure = CudaEventFailureInjection::kSynchronize;
    }
    auto work = std::make_shared<CudaWork>(
        py::cast(output), token, lease, event_failure);
    side_effects_.mark_work_publish();
    return work;
  } catch (...) {
    if (workspace_access_started && lease) {
      lease->quarantine();
    }
    throw;
  }
}

void ReducedShardPlan::exhaust_sequence_for_test() {
  next_sequence_.exhaust_for_test();
}

void ReducedShardPlan::inject_failure_for_test(
    const std::string& failure) {
  failure_for_test_ = parse_failure_injection_for_test(failure);
}

void ReducedShardPlan::arm_dequant_gate_for_test() {
  bool expected = false;
  if (!test_delay_armed_.compare_exchange_strong(
          expected,
          true,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    throw py::value_error("ReducedShard test delay is already armed");
  }
}

uint64_t ReducedShardPlan::test_delay_launch_count_for_test() const noexcept {
  return test_delay_launch_count_.load(std::memory_order_relaxed);
}

py::dict ReducedShardPlan::side_effect_counts_for_test() const {
  py::dict counts;
  counts["allocation"] = side_effects_.allocation();
  counts["workspace_acquire"] = side_effects_.workspace_acquire();
  counts["transport_launch"] = side_effects_.transport_launch();
  counts["kernel_launch"] = side_effects_.kernel_launch();
  counts["work_publish"] = side_effects_.work_publish();
  return counts;
}

std::shared_ptr<CudaWork> ReducedShardPlan::execute(
    torch::Tensor input,
    c10::optional<torch::Tensor> committed_residual,
    torch::Tensor* candidate_residual) {
  try {
    validate_input(input, committed_residual);
  } catch (const CudaExecutionError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaExecutionError(
        std::string("ReducedShard execution failed: ") + error.what());
  } catch (...) {
    throw CudaExecutionError("ReducedShard execution failed");
  }
  const LaunchToken token{
      plan_id_, allocate_cuda_sequence(next_sequence_)};
  try {
    if (compression_ == ReducedShardCompression::kNative) {
      return execute_native(std::move(input), token);
    }
    return execute_int8(
        std::move(input),
        token,
        committed_residual,
        candidate_residual);
  } catch (const CudaExecutionError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaExecutionError(
        std::string("ReducedShard execution failed: ") + error.what());
  } catch (...) {
    throw CudaExecutionError("ReducedShard execution failed");
  }
}

std::shared_ptr<ReducedShardPlan> create_reduced_shard_plan(
    py::dict config,
    py::object process_group) {
  validate_exact_keys(config);
  const ReducedShardCompression compression = parse_compression(config);
  const ReducedShardReduction reduction = parse_reduction(config);
  const at::ScalarType dtype = parse_dtype(config);
  const int64_t numel = exact_nonnegative_int(config, "numel");
  const int64_t rank = exact_nonnegative_int(config, "rank");
  const int64_t world_size = exact_nonnegative_int(config, "world_size");
  if (world_size != 2 && world_size != 4) {
    throw py::value_error("CUDA ProcessGroup world size is unsupported");
  }
  if (rank >= world_size) {
    throw py::value_error("CUDA ProcessGroup rank is invalid");
  }
  const int64_t logical_shard_length =
      checked_add(numel, world_size - 1, "logical shard") / world_size;
  validate_common_layout(
      config, numel, rank, world_size, logical_shard_length);
  if (compression == ReducedShardCompression::kNative) {
    validate_native_layout(
        config, numel, world_size, logical_shard_length);
  } else {
    validate_int8_layout(config, world_size, logical_shard_length);
  }
  const int64_t group_size =
      compression == ReducedShardCompression::kInt8
      ? exact_nonnegative_int(config, "group_size")
      : 0;
  const bool gradient_error_feedback =
      parse_gradient_error_feedback(config);
  if (gradient_error_feedback &&
      (compression != ReducedShardCompression::kInt8 || group_size != 64)) {
    throw py::value_error(
        "CUDA ReducedShard gradient error feedback requires INT8 group size 64");
  }
  const int64_t valid_length =
      exact_nonnegative_int(config, "valid_length");
  const int64_t transport_shard_length =
      exact_nonnegative_int(config, "transport_shard_length");
  const int64_t payload_bytes_per_destination =
      exact_nonnegative_int(config, "payload_bytes_per_destination");
  const int64_t send_payload_bytes =
      exact_nonnegative_int(config, "send_payload_bytes");
  const int64_t receive_payload_bytes =
      exact_nonnegative_int(config, "receive_payload_bytes");
  const int64_t workspace_bytes =
      exact_nonnegative_int(config, "workspace_bytes");

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
    throw py::value_error("CUDA ReducedShard execution requires NCCL");
  }
  if (group->getRank() != rank) {
    throw py::value_error("CUDA ProcessGroup rank mismatch");
  }
  if (group->getSize() != world_size) {
    throw py::value_error("CUDA ProcessGroup world size mismatch");
  }
  return std::make_shared<ReducedShardPlan>(
      compression,
      reduction,
      dtype,
      numel,
      logical_shard_length,
      valid_length,
      rank,
      world_size,
      group_size,
      transport_shard_length,
      payload_bytes_per_destination,
      send_payload_bytes,
      receive_payload_bytes,
      workspace_bytes,
      gradient_error_feedback,
      std::move(group));
}

void bind_reduced_shard_plan(py::module_& module) {
  py::class_<ReducedShardPlan, std::shared_ptr<ReducedShardPlan>>(
      module, "ReducedShardPlan", py::dynamic_attr())
      .def(
          "_exhaust_sequence_for_test",
          &ReducedShardPlan::exhaust_sequence_for_test)
      .def(
          "_inject_failure_for_test",
          &ReducedShardPlan::inject_failure_for_test)
      .def(
          "_arm_dequant_gate_for_test",
          &ReducedShardPlan::arm_dequant_gate_for_test)
      .def(
          "_test_delay_launch_count_for_test",
          &ReducedShardPlan::test_delay_launch_count_for_test)
      .def(
          "_side_effect_counts_for_test",
          &ReducedShardPlan::side_effect_counts_for_test);
  module.def(
      "create_reduced_shard_plan",
      [](py::dict config, py::object process_group) {
        std::shared_ptr<ReducedShardPlan> plan =
            create_reduced_shard_plan(
                std::move(config), std::move(process_group));
        py::object result = py::cast(plan);
        result.attr("execute") = py::cpp_function(
            [plan](torch::Tensor input, py::object residual) {
              c10::optional<torch::Tensor> committed_residual = c10::nullopt;
              if (!residual.is_none()) {
                committed_residual = residual.cast<torch::Tensor>();
              }
              torch::Tensor candidate_residual;
              std::shared_ptr<CudaWork> work =
                  plan->execute(
                      std::move(input),
                      std::move(committed_residual),
                      &candidate_residual);
              py::object work_object = py::cast(work);
              work_object.attr("is_completed") = py::cpp_function(
                  [work]() { return work->is_completed(); });
              work_object.attr("wait") = py::cpp_function(
                  [work]() { return work->wait(); });
              work_object.attr("result") = py::cpp_function(
                  [work]() { return work->result(); });
              work_object.attr("launch_token") = py::cpp_function(
                  [work]() { return work->launch_token(); });
              if (candidate_residual.defined()) {
                work_object.attr("_candidate_gradient_residual") =
                    py::cpp_function(
                        [candidate_residual]() {
                          return candidate_residual;
                        });
              }
              return work_object;
            },
            py::arg("input"),
            py::arg("committed_residual") = py::none());
        return result;
      },
      py::arg("config"),
      py::arg("process_group"));
}

}  // namespace ccdl_comm
