#include "qwd_plan.h"

#include "../quantization/dequant_api.cuh"
#include "../quantization/quant_api.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

namespace ccdl_comm {

namespace {

constexpr std::array<const char*, 17> kConfigKeys{
    "accumulation_dtype",
    "collective",
    "compression",
    "dtype",
    "fp32_gathered_bytes",
    "global_numel",
    "group_size",
    "groups_per_shard",
    "output_bytes",
    "payload_bytes_per_rank",
    "qwd_gathered_payload_bytes",
    "rank",
    "shard_numel",
    "start",
    "valid_numel",
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
        std::string("CUDA config field exceeds signed 64-bit range: ") + key);
  }
  if (converted < 0) {
    throw py::value_error(
        std::string("CUDA config field must be non-negative: ") + key);
  }
  return static_cast<int64_t>(converted);
}

int64_t checked_add(int64_t left, int64_t right, const char* label) {
  if (right > std::numeric_limits<int64_t>::max() - left) {
    throw py::value_error(std::string("CUDA qWD ") + label + " size overflow");
  }
  return left + right;
}

int64_t checked_mul(int64_t left, int64_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<int64_t>::max() / left) {
    throw py::value_error(std::string("CUDA qWD ") + label + " size overflow");
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
  if (config.size() != kConfigKeys.size()) {
    throw py::value_error("CUDA config fields are invalid");
  }
}

void validate_descriptor(
    const py::dict& config,
    int64_t global_numel,
    int64_t rank,
    int64_t world_size,
    int64_t shard_numel) {
  if (exact_string(config, "dtype") != "fp16") {
    throw py::value_error("CUDA qWD dtype must be fp16");
  }
  if (exact_string(config, "accumulation_dtype") != "fp32") {
    throw py::value_error("CUDA qWD accumulation must be fp32");
  }
  if (exact_string(config, "compression") != "int8") {
    throw py::value_error("CUDA qWD compression must be int8");
  }
  if (exact_string(config, "collective") != "all_gather") {
    throw py::value_error("CUDA qWD collective must be all_gather");
  }
  if (exact_nonnegative_int(config, "group_size") != 64) {
    throw py::value_error("CUDA qWD group size must be 64");
  }

  const int64_t expected_shard = global_numel / world_size +
      (global_numel % world_size != 0 ? 1 : 0);
  if (shard_numel != expected_shard) {
    throw py::value_error("CUDA qWD shard layout is inconsistent");
  }
  const int64_t raw_start = checked_mul(rank, shard_numel, "ownership");
  const int64_t expected_start = std::min(raw_start, global_numel);
  const int64_t expected_valid =
      std::min(shard_numel, global_numel - expected_start);
  if (exact_nonnegative_int(config, "start") != expected_start ||
      exact_nonnegative_int(config, "valid_numel") != expected_valid) {
    throw py::value_error("CUDA qWD ownership is inconsistent");
  }

  const int64_t groups = shard_numel / 64 + (shard_numel % 64 != 0 ? 1 : 0);
  const int64_t payload = checked_mul(groups, 68, "payload");
  const int64_t gathered_payload =
      checked_mul(payload, world_size, "gathered payload");
  const int64_t fp32_gathered = checked_mul(
      checked_mul(shard_numel, world_size, "refresh"), 4, "refresh");
  const int64_t output_bytes = checked_mul(
      checked_mul(shard_numel, world_size, "output"), 2, "output");
  const int64_t qwd_workspace =
      checked_add(payload, gathered_payload, "workspace");
  const int64_t workspace = std::max(qwd_workspace, fp32_gathered);
  if (exact_nonnegative_int(config, "groups_per_shard") != groups ||
      exact_nonnegative_int(config, "payload_bytes_per_rank") != payload ||
      exact_nonnegative_int(config, "qwd_gathered_payload_bytes") !=
          gathered_payload ||
      exact_nonnegative_int(config, "fp32_gathered_bytes") != fp32_gathered) {
    throw py::value_error("CUDA qWD payload layout is inconsistent");
  }
  if (exact_nonnegative_int(config, "output_bytes") != output_bytes) {
    throw py::value_error("CUDA qWD output layout is inconsistent");
  }
  if (exact_nonnegative_int(config, "workspace_bytes") != workspace) {
    throw py::value_error("CUDA qWD workspace is inconsistent");
  }
}

QWDFailureInjection parse_failure_injection_for_test(
    const std::string& failure) {
  if (failure == "none") return QWDFailureInjection::kNone;
  if (failure == "quant_helper") return QWDFailureInjection::kQuantHelper;
  if (failure == "transport_wait") return QWDFailureInjection::kTransportWait;
  if (failure == "restore_helper") return QWDFailureInjection::kRestoreHelper;
  if (failure == "event_record") return QWDFailureInjection::kEventRecord;
  if (failure == "event_synchronize") {
    return QWDFailureInjection::kEventSynchronize;
  }
  throw py::value_error("qWD failure injection is invalid");
}

}  // namespace

QWDPlan::QWDPlan(
    int64_t global_numel,
    int64_t shard_numel,
    int64_t start,
    int64_t valid_numel,
    int64_t rank,
    int64_t world_size,
    int64_t groups_per_shard,
    int64_t payload_bytes_per_rank,
    int64_t qwd_gathered_payload_bytes,
    int64_t fp32_gathered_bytes,
    int64_t workspace_bytes,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group)
    : global_numel_(global_numel),
      shard_numel_(shard_numel),
      start_(start),
      valid_numel_(valid_numel),
      rank_(rank),
      world_size_(world_size),
      groups_per_shard_(groups_per_shard),
      payload_bytes_per_rank_(payload_bytes_per_rank),
      qwd_gathered_payload_bytes_(qwd_gathered_payload_bytes),
      fp32_gathered_bytes_(fp32_gathered_bytes),
      workspace_bytes_(workspace_bytes),
      process_group_(std::move(process_group)),
      workspace_pool_(std::make_shared<WorkspacePool>(workspace_bytes)),
      plan_id_(allocate_cuda_plan_id()) {}

void QWDPlan::validate_inputs(
    const torch::Tensor& master_shard,
    const torch::Tensor& model_copy_flat,
    const std::string& mode) const {
  if (mode != "qwd" && mode != "fp_refresh") {
    throw CudaExecutionError("qWD mode must be qwd or fp_refresh");
  }
  if (!master_shard.defined() || !master_shard.is_cuda()) {
    throw CudaExecutionError("qWD master_shard must be a CUDA tensor");
  }
  if (!master_shard.is_contiguous()) {
    throw CudaExecutionError("qWD master_shard must be contiguous");
  }
  if (master_shard.scalar_type() != at::kFloat) {
    throw CudaExecutionError("qWD master_shard dtype must be fp32");
  }
  if (master_shard.numel() != shard_numel_) {
    throw CudaExecutionError("qWD master_shard numel mismatch");
  }
  if (!model_copy_flat.defined() || !model_copy_flat.is_cuda()) {
    throw CudaExecutionError("qWD model_copy_flat must be a CUDA tensor");
  }
  if (!model_copy_flat.is_contiguous()) {
    throw CudaExecutionError("qWD model_copy_flat must be contiguous");
  }
  if (model_copy_flat.scalar_type() != at::kHalf) {
    throw CudaExecutionError("qWD model_copy_flat dtype must be fp16");
  }
  if (model_copy_flat.numel() != shard_numel_ * world_size_) {
    throw CudaExecutionError("qWD model_copy_flat numel mismatch");
  }
  if (model_copy_flat.device() != master_shard.device()) {
    throw CudaExecutionError("qWD inputs must be on the same CUDA device");
  }
  if (master_shard.is_alias_of(model_copy_flat)) {
    throw CudaExecutionError("qWD inputs must not alias storage");
  }
}

std::shared_ptr<CudaWork> QWDPlan::execute_qwd(
    const torch::Tensor& master_shard,
    const torch::Tensor& model_copy_flat,
    torch::Tensor output,
    LaunchToken token) {
  if (shard_numel_ == 0) {
    side_effects_.mark_work_publish();
    return std::make_shared<CudaWork>(py::cast(output), token);
  }
  side_effects_.mark_workspace_acquire();
  std::unique_ptr<WorkspaceLease> lease =
      workspace_pool_->acquire(workspace_bytes_);
  const torch::Tensor& storage = lease->storage();
  torch::Tensor send = storage.narrow(0, 0, payload_bytes_per_rank_);
  torch::Tensor gathered = storage.narrow(
      0, payload_bytes_per_rank_, qwd_gathered_payload_bytes_);
  const int64_t model_shard_offset = rank_ * shard_numel_;
  torch::Tensor model_shard = model_copy_flat.narrow(
      0, model_shard_offset, shard_numel_);
  bool workspace_access_started = false;
  try {
    side_effects_.mark_kernel_launch();
    workspace_access_started = true;
    if (!try_inplace_quantize_parameter_delta(
            master_shard, model_shard, send, valid_numel_, 64)) {
      throw CudaExecutionError("qWD quantize helper is unsupported");
    }
    if (failure_for_test_ == QWDFailureInjection::kQuantHelper) {
      throw CudaExecutionError("test injected quant_helper failure");
    }

    side_effects_.mark_transport_launch();
    c10::intrusive_ptr<c10d::Work> transport =
        process_group_->_allgather_base(
            gathered, send, c10d::AllgatherOptions{});
    if (!transport) {
      throw CudaExecutionError("NCCL qWD all-gather returned no Work");
    }
    if (failure_for_test_ == QWDFailureInjection::kTransportWait) {
      throw CudaExecutionError("test injected transport_wait failure");
    }
    bool completed = false;
    {
      py::gil_scoped_release release;
      completed = transport->wait();
    }
    if (!completed) {
      throw CudaExecutionError("NCCL qWD all-gather did not complete");
    }

    side_effects_.mark_kernel_launch();
    if (!try_inplace_dequantize_gathered_add(
            gathered,
            output,
            world_size_,
            payload_bytes_per_rank_,
            shard_numel_,
            global_numel_)) {
      throw CudaExecutionError("qWD gathered-add helper is unsupported");
    }
    if (failure_for_test_ == QWDFailureInjection::kRestoreHelper) {
      throw CudaExecutionError("test injected restore_helper failure");
    }
    CudaEventFailureInjection event_failure =
        CudaEventFailureInjection::kNone;
    if (failure_for_test_ == QWDFailureInjection::kEventRecord) {
      event_failure = CudaEventFailureInjection::kRecord;
    } else if (
        failure_for_test_ == QWDFailureInjection::kEventSynchronize) {
      event_failure = CudaEventFailureInjection::kSynchronize;
    }
    auto work = std::make_shared<CudaWork>(
        py::cast(output), token, lease, event_failure);
    side_effects_.mark_work_publish();
    return work;
  } catch (...) {
    if (workspace_access_started && lease) lease->quarantine();
    throw;
  }
}

std::shared_ptr<CudaWork> QWDPlan::execute_refresh(
    const torch::Tensor& master_shard,
    torch::Tensor output,
    LaunchToken token) {
  if (shard_numel_ == 0) {
    side_effects_.mark_work_publish();
    return std::make_shared<CudaWork>(py::cast(output), token);
  }
  side_effects_.mark_workspace_acquire();
  std::unique_ptr<WorkspaceLease> lease =
      workspace_pool_->acquire(workspace_bytes_);
  const torch::Tensor& storage = lease->storage();
  torch::Tensor gathered = torch::from_blob(
      storage.data_ptr(),
      {shard_numel_ * world_size_},
      storage.options().dtype(torch::kFloat32));
  bool workspace_access_started = false;
  try {
    torch::Tensor send = master_shard;
    workspace_access_started = true;
    side_effects_.mark_transport_launch();
    c10::intrusive_ptr<c10d::Work> transport =
        process_group_->_allgather_base(
            gathered, send, c10d::AllgatherOptions{});
    if (!transport) {
      throw CudaExecutionError("NCCL refresh all-gather returned no Work");
    }
    if (failure_for_test_ == QWDFailureInjection::kTransportWait) {
      throw CudaExecutionError("test injected transport_wait failure");
    }
    bool completed = false;
    {
      py::gil_scoped_release release;
      completed = transport->wait();
    }
    if (!completed) {
      throw CudaExecutionError("NCCL refresh all-gather did not complete");
    }
    side_effects_.mark_kernel_launch();
    if (!try_inplace_qwd_refresh_cast(gathered, output)) {
      throw CudaExecutionError("qWD refresh cast helper is unsupported");
    }
    if (failure_for_test_ == QWDFailureInjection::kRestoreHelper) {
      throw CudaExecutionError("test injected restore_helper failure");
    }
    CudaEventFailureInjection event_failure =
        CudaEventFailureInjection::kNone;
    if (failure_for_test_ == QWDFailureInjection::kEventRecord) {
      event_failure = CudaEventFailureInjection::kRecord;
    } else if (
        failure_for_test_ == QWDFailureInjection::kEventSynchronize) {
      event_failure = CudaEventFailureInjection::kSynchronize;
    }
    auto work = std::make_shared<CudaWork>(
        py::cast(output), token, lease, event_failure);
    side_effects_.mark_work_publish();
    return work;
  } catch (...) {
    if (workspace_access_started && lease) lease->quarantine();
    throw;
  }
}

std::shared_ptr<CudaWork> QWDPlan::execute(
    torch::Tensor master_shard,
    torch::Tensor model_copy_flat,
    const std::string& mode) {
  try {
    validate_inputs(master_shard, model_copy_flat, mode);
  } catch (const CudaExecutionError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaExecutionError(std::string("qWD execution failed: ") + error.what());
  }
  const LaunchToken token{plan_id_, allocate_cuda_sequence(next_sequence_)};
  c10::cuda::CUDAGuard device_guard(master_shard.device());
  side_effects_.mark_allocation();
  torch::Tensor output = mode == "qwd"
      ? model_copy_flat.clone()
      : torch::empty_like(model_copy_flat);
  try {
    if (mode == "qwd") {
      return execute_qwd(master_shard, model_copy_flat, output, token);
    }
    return execute_refresh(master_shard, output, token);
  } catch (const CudaExecutionError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaExecutionError(std::string("qWD execution failed: ") + error.what());
  } catch (...) {
    throw CudaExecutionError("qWD execution failed");
  }
}

void QWDPlan::exhaust_sequence_for_test() {
  next_sequence_.exhaust_for_test();
}

void QWDPlan::inject_failure_for_test(const std::string& failure) {
  failure_for_test_ = parse_failure_injection_for_test(failure);
}

py::dict QWDPlan::side_effect_counts_for_test() const {
  py::dict counts;
  counts["allocation"] = side_effects_.allocation();
  counts["workspace_acquire"] = side_effects_.workspace_acquire();
  counts["transport_launch"] = side_effects_.transport_launch();
  counts["kernel_launch"] = side_effects_.kernel_launch();
  counts["work_publish"] = side_effects_.work_publish();
  return counts;
}

std::shared_ptr<QWDPlan> create_qwd_plan(
    py::dict config,
    py::object process_group) {
  validate_exact_keys(config);
  const int64_t global_numel = exact_nonnegative_int(config, "global_numel");
  const int64_t rank = exact_nonnegative_int(config, "rank");
  const int64_t world_size = exact_nonnegative_int(config, "world_size");
  if (world_size != 2 && world_size != 4) {
    throw py::value_error("CUDA ProcessGroup world size is unsupported");
  }
  if (rank >= world_size) {
    throw py::value_error("CUDA ProcessGroup rank is invalid");
  }
  const int64_t shard_numel = exact_nonnegative_int(config, "shard_numel");
  validate_descriptor(config, global_numel, rank, world_size, shard_numel);

  c10::intrusive_ptr<c10d::ProcessGroup> group;
  try {
    group = process_group.cast<c10::intrusive_ptr<c10d::ProcessGroup>>();
  } catch (const py::cast_error& error) {
    throw py::value_error(
        std::string("CUDA qWD requires a c10d ProcessGroupNCCL: ") +
        error.what());
  }
  c10::intrusive_ptr<c10d::Backend> cuda_backend;
  if (group) {
    try {
      cuda_backend = group->getBackend(c10::DeviceType::CUDA);
    } catch (...) {
      cuda_backend.reset();
    }
  }
  if (!cuda_backend ||
      dynamic_cast<c10d::ProcessGroupNCCL*>(cuda_backend.get()) == nullptr) {
    throw py::value_error("CUDA qWD requires a c10d ProcessGroupNCCL");
  }
  if (group->getRank() != rank) {
    throw py::value_error("CUDA ProcessGroup rank mismatch");
  }
  if (group->getSize() != world_size) {
    throw py::value_error("CUDA ProcessGroup world size mismatch");
  }
  return std::make_shared<QWDPlan>(
      global_numel,
      shard_numel,
      exact_nonnegative_int(config, "start"),
      exact_nonnegative_int(config, "valid_numel"),
      rank,
      world_size,
      exact_nonnegative_int(config, "groups_per_shard"),
      exact_nonnegative_int(config, "payload_bytes_per_rank"),
      exact_nonnegative_int(config, "qwd_gathered_payload_bytes"),
      exact_nonnegative_int(config, "fp32_gathered_bytes"),
      exact_nonnegative_int(config, "workspace_bytes"),
      std::move(group));
}

void bind_qwd_plan(py::module_& module) {
  py::class_<QWDPlan, std::shared_ptr<QWDPlan>>(module, "_QWDPlan")
      .def("execute", &QWDPlan::execute)
      .def("_exhaust_sequence_for_test", &QWDPlan::exhaust_sequence_for_test)
      .def("_inject_failure_for_test", &QWDPlan::inject_failure_for_test)
      .def("_side_effect_counts_for_test", &QWDPlan::side_effect_counts_for_test);
  module.def(
      "_create_qwd_plan",
      &create_qwd_plan,
      py::arg("config"),
      py::arg("process_group"));
}

}  // namespace ccdl_comm
