#pragma once

#include "compressed_work.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ccdl_comm {

class CudaExecutor {
 public:
  explicit CudaExecutor(size_t workspace_capacity_bytes);
  std::shared_ptr<CudaWork> run(
      py::object result,
      size_t workspace_bytes = 0);
  void exhaust_sequence_for_test();
  void inject_event_failure_for_test(const std::string& failure);
  py::dict side_effect_counts_for_test() const;

 private:
  uint64_t plan_id_;
  SaturatingMonotonicAllocator next_sequence_{1};
  LaunchSideEffectCounters side_effects_;
  std::shared_ptr<WorkspacePool> workspace_pool_;
  CudaEventFailureInjection event_failure_for_test_{
      CudaEventFailureInjection::kNone};
};

std::shared_ptr<CudaExecutor> create_cuda_executor(
    size_t workspace_capacity_bytes = 0);
void bind_cuda_executor(py::module_& module);

}  // namespace ccdl_comm
