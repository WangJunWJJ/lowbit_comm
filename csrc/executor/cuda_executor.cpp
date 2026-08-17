#include "cuda_executor.h"

#include <utility>

namespace ccdl_comm {

namespace {

std::atomic<uint64_t> next_plan_id{1};

}  // namespace

CudaExecutor::CudaExecutor(size_t workspace_capacity_bytes)
    : plan_id_(next_plan_id.fetch_add(1, std::memory_order_relaxed)),
      workspace_pool_(
          std::make_shared<WorkspacePool>(workspace_capacity_bytes)) {}

std::shared_ptr<CudaWork> CudaExecutor::run(
    py::object result,
    size_t workspace_bytes) {
  std::unique_ptr<WorkspaceLease> lease;
  if (workspace_bytes != 0) {
    lease = workspace_pool_->acquire(workspace_bytes);
  }
  const LaunchToken token{
      plan_id_, next_sequence_.fetch_add(1, std::memory_order_relaxed)};
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
