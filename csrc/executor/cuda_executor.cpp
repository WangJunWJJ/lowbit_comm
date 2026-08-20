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
  std::unique_ptr<WorkspaceLease> lease;
  if (workspace_bytes != 0) {
    lease = workspace_pool_->acquire(workspace_bytes);
  }
  const LaunchToken token{plan_id_, allocate_cuda_sequence(next_sequence_)};
  return std::make_shared<CudaWork>(
      std::move(result), token, std::move(lease));
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
          py::arg("workspace_bytes") = 0);
  module.def(
      "create_cuda_executor",
      &create_cuda_executor,
      py::arg("workspace_capacity_bytes") = 0);
}

}  // namespace ccdl_comm
