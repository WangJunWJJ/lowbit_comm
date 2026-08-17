#pragma once

#include "compressed_work.h"

#include <atomic>
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

 private:
  uint64_t plan_id_;
  std::atomic<uint64_t> next_sequence_{1};
  std::shared_ptr<WorkspacePool> workspace_pool_;
};

std::shared_ptr<CudaExecutor> create_cuda_executor(
    size_t workspace_capacity_bytes = 0);
void bind_cuda_executor(py::module_& module);

}  // namespace ccdl_comm
