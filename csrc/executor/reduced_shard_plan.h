#pragma once

#include "compressed_work.h"

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <atomic>
#include <cstdint>
#include <memory>

namespace ccdl_comm {

enum class ReducedShardCompression : uint8_t {
  kNative,
  kInt8,
};

enum class ReducedShardReduction : uint8_t {
  kSum,
  kMean,
};

enum class ReducedShardFailureInjection : uint8_t {
  kNone,
  kQuantHelper,
  kTransportWait,
  kDequantHelper,
  kEventRecord,
  kEventSynchronize,
};

class ReducedShardPlan {
 public:
  ReducedShardPlan(
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
      c10::intrusive_ptr<c10d::ProcessGroup> process_group);

  std::shared_ptr<CudaWork> execute(
      torch::Tensor input,
      c10::optional<torch::Tensor> committed_residual,
      torch::Tensor* candidate_residual);
  void exhaust_sequence_for_test();
  void inject_failure_for_test(const std::string& failure);
  void arm_dequant_gate_for_test();
  uint64_t test_delay_launch_count_for_test() const noexcept;
  py::dict side_effect_counts_for_test() const;

 private:
  void validate_input(
      const torch::Tensor& input,
      const c10::optional<torch::Tensor>& committed_residual) const;
  std::shared_ptr<CudaWork> execute_native(
      torch::Tensor input,
      LaunchToken token);
  std::shared_ptr<CudaWork> execute_int8(
      torch::Tensor input,
      LaunchToken token,
      const c10::optional<torch::Tensor>& committed_residual,
      torch::Tensor* candidate_residual);

  ReducedShardCompression compression_;
  ReducedShardReduction reduction_;
  at::ScalarType dtype_;
  int64_t numel_;
  int64_t logical_shard_length_;
  int64_t valid_length_;
  int64_t rank_;
  int64_t world_size_;
  int64_t group_size_;
  int64_t transport_shard_length_;
  int64_t payload_bytes_per_destination_;
  int64_t send_payload_bytes_;
  int64_t receive_payload_bytes_;
  int64_t workspace_bytes_;
  bool gradient_error_feedback_;
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
  std::shared_ptr<WorkspacePool> workspace_pool_;
  uint64_t plan_id_;
  SaturatingMonotonicAllocator next_sequence_{1};
  LaunchSideEffectCounters side_effects_;
  ReducedShardFailureInjection failure_for_test_{
      ReducedShardFailureInjection::kNone};
  std::atomic<bool> test_delay_armed_{false};
  std::atomic<uint64_t> test_delay_launch_count_{0};
};

std::shared_ptr<ReducedShardPlan> create_reduced_shard_plan(
    py::dict config,
    py::object process_group);
void bind_reduced_shard_plan(py::module_& module);

}  // namespace ccdl_comm
