#pragma once

#include "compressed_work.h"

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace ccdl_comm {

enum class QWDFailureInjection : uint8_t {
  kNone,
  kQuantHelper,
  kTransportWait,
  kRestoreHelper,
  kEventRecord,
  kEventSynchronize,
};

class QWDPlan final {
 public:
  QWDPlan(
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
      c10::intrusive_ptr<c10d::ProcessGroup> process_group);

  std::shared_ptr<CudaWork> execute(
      torch::Tensor master_shard,
      torch::Tensor model_copy_flat,
      const std::string& mode);
  void exhaust_sequence_for_test();
  void inject_failure_for_test(const std::string& failure);
  py::dict side_effect_counts_for_test() const;

 private:
  void validate_inputs(
      const torch::Tensor& master_shard,
      const torch::Tensor& model_copy_flat,
      const std::string& mode) const;
  std::shared_ptr<CudaWork> execute_qwd(
      const torch::Tensor& master_shard,
      const torch::Tensor& model_copy_flat,
      torch::Tensor output,
      LaunchToken token);
  std::shared_ptr<CudaWork> execute_refresh(
      const torch::Tensor& master_shard,
      torch::Tensor output,
      LaunchToken token);

  int64_t global_numel_;
  int64_t shard_numel_;
  int64_t start_;
  int64_t valid_numel_;
  int64_t rank_;
  int64_t world_size_;
  int64_t groups_per_shard_;
  int64_t payload_bytes_per_rank_;
  int64_t qwd_gathered_payload_bytes_;
  int64_t fp32_gathered_bytes_;
  int64_t workspace_bytes_;
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
  std::shared_ptr<WorkspacePool> workspace_pool_;
  uint64_t plan_id_;
  SaturatingMonotonicAllocator next_sequence_{1};
  LaunchSideEffectCounters side_effects_;
  QWDFailureInjection failure_for_test_{QWDFailureInjection::kNone};
};

std::shared_ptr<QWDPlan> create_qwd_plan(
    py::dict config,
    py::object process_group);
void bind_qwd_plan(py::module_& module);

}  // namespace ccdl_comm
