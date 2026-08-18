#pragma once

#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace py = pybind11;

namespace ccdl_comm {

class WorkspacePool;

class WorkspaceLease {
 public:
  WorkspaceLease(
      std::shared_ptr<WorkspacePool> pool,
      uint64_t lease_id,
      size_t size_bytes,
      torch::Tensor storage);
  ~WorkspaceLease();

  WorkspaceLease(const WorkspaceLease&) = delete;
  WorkspaceLease& operator=(const WorkspaceLease&) = delete;
  WorkspaceLease(WorkspaceLease&&) = delete;
  WorkspaceLease& operator=(WorkspaceLease&&) = delete;

  uint64_t lease_id() const;
  const torch::Tensor& storage() const noexcept;
  void release();

 private:
  std::shared_ptr<WorkspacePool> pool_;
  uint64_t lease_id_;
  size_t size_bytes_;
  torch::Tensor storage_;
  bool released_{false};
};

class WorkspacePool : public std::enable_shared_from_this<WorkspacePool> {
 public:
  explicit WorkspacePool(size_t capacity_bytes);
  std::unique_ptr<WorkspaceLease> acquire(size_t size_bytes);
  void release(size_t size_bytes);

 private:
  size_t capacity_bytes_;
  size_t used_bytes_{0};
  uint64_t next_lease_id_{1};
  std::mutex mutex_;
};

class TestWorkspaceLease {
 public:
  explicit TestWorkspaceLease(std::unique_ptr<WorkspaceLease> lease);
  uint64_t lease_id() const;
  void complete_for_test();

 private:
  uint64_t lease_id_;
  std::unique_ptr<WorkspaceLease> lease_;
};

void bind_workspace_pool(py::module_& module);

}  // namespace ccdl_comm
