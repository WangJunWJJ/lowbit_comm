#include "cuda_executor.h"

#include <utility>

namespace ccdl_comm {

CudaExecutor::CudaExecutor(size_t workspace_capacity_bytes)
    : plan_id_(allocate_cuda_plan_id()),
      workspace_pool_(
          std::make_shared<WorkspacePool>(workspace_capacity_bytes)) {}

std::shared_ptr<CudaWork> CudaExecutor::run(
    py::object result,
    size_t workspace_bytes) {
  const LaunchToken token{plan_id_, allocate_cuda_sequence(next_sequence_)};
  std::unique_ptr<WorkspaceLease> lease;
  if (workspace_bytes != 0) {
    side_effects_.mark_workspace_acquire();
    lease = workspace_pool_->acquire(workspace_bytes);
  }
  side_effects_.mark_work_publish();
  if (lease) {
    return std::make_shared<CudaWork>(
        std::move(result), token, lease, event_failure_for_test_);
  }
  return std::make_shared<CudaWork>(std::move(result), token);
}

void CudaExecutor::exhaust_sequence_for_test() {
  next_sequence_.exhaust_for_test();
}

void CudaExecutor::inject_event_failure_for_test(
    const std::string& failure) {
  event_failure_for_test_ =
      parse_cuda_event_failure_injection_for_test(failure);
}

py::dict CudaExecutor::side_effect_counts_for_test() const {
  py::dict counts;
  counts["allocation"] = side_effects_.allocation();
  counts["workspace_acquire"] = side_effects_.workspace_acquire();
  counts["transport_launch"] = side_effects_.transport_launch();
  counts["kernel_launch"] = side_effects_.kernel_launch();
  counts["work_publish"] = side_effects_.work_publish();
  return counts;
}

std::shared_ptr<CudaExecutor> create_cuda_executor(
    size_t workspace_capacity_bytes) {
  return std::make_shared<CudaExecutor>(workspace_capacity_bytes);
}

void bind_cuda_executor(py::module_& module) {
  py::class_<CudaExecutor, std::shared_ptr<CudaExecutor>>(
      module, "CudaExecutor")
      .def(
          "run",
          &CudaExecutor::run,
          py::arg("result"),
          py::arg("workspace_bytes") = 0)
      .def(
          "_exhaust_sequence_for_test",
          &CudaExecutor::exhaust_sequence_for_test)
      .def(
          "_inject_event_failure_for_test",
          &CudaExecutor::inject_event_failure_for_test)
      .def(
          "_side_effect_counts_for_test",
          &CudaExecutor::side_effect_counts_for_test);
  module.def(
      "create_cuda_executor",
      &create_cuda_executor,
      py::arg("workspace_capacity_bytes") = 0);
}

}  // namespace ccdl_comm
