#include "workspace_pool.h"

#include "../executor/compressed_work.h"

#include <utility>

namespace ccdl_comm {

WorkspaceLease::WorkspaceLease(
    std::shared_ptr<WorkspacePool> pool,
    uint64_t lease_id,
    size_t size_bytes,
    torch::Tensor storage)
    : pool_(std::move(pool)),
      lease_id_(lease_id),
      size_bytes_(size_bytes),
      storage_(std::move(storage)) {}

WorkspaceLease::~WorkspaceLease() {
  release();
}

uint64_t WorkspaceLease::lease_id() const {
  return lease_id_;
}

const torch::Tensor& WorkspaceLease::storage() const noexcept {
  return storage_;
}

void WorkspaceLease::release() {
  if (released_) {
    return;
  }
  released_ = true;
  storage_.reset();
  pool_->release(size_bytes_);
}

WorkspacePool::WorkspacePool(size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes) {}

std::unique_ptr<WorkspaceLease> WorkspacePool::acquire(size_t size_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (size_bytes > capacity_bytes_ - used_bytes_) {
    throw CudaExecutionError("workspace pool capacity exhausted");
  }
  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .device(torch::kCUDA);
  torch::Tensor storage = torch::empty(
      {static_cast<int64_t>(size_bytes)}, options);
  const uint64_t lease_id = next_lease_id_++;
  used_bytes_ += size_bytes;
  return std::make_unique<WorkspaceLease>(
      shared_from_this(), lease_id, size_bytes, std::move(storage));
}

void WorkspacePool::release(size_t size_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  used_bytes_ -= size_bytes;
}

TestWorkspaceLease::TestWorkspaceLease(
    std::unique_ptr<WorkspaceLease> lease)
    : lease_id_(lease->lease_id()), lease_(std::move(lease)) {}

uint64_t TestWorkspaceLease::lease_id() const {
  return lease_id_;
}

void TestWorkspaceLease::complete_for_test() {
  lease_.reset();
}

namespace {

std::shared_ptr<WorkspacePool> test_pool;

void reset_test_workspace_pool(size_t capacity_bytes) {
  test_pool = std::make_shared<WorkspacePool>(capacity_bytes);
}

std::shared_ptr<TestWorkspaceLease> acquire_test_lease(size_t size_bytes) {
  if (!test_pool) {
    throw CudaExecutionError("workspace pool is not initialized");
  }
  return std::make_shared<TestWorkspaceLease>(test_pool->acquire(size_bytes));
}

}  // namespace

void bind_workspace_pool(py::module_& module) {
  py::class_<TestWorkspaceLease, std::shared_ptr<TestWorkspaceLease>>(
      module, "_TestWorkspaceLease")
      .def_property_readonly("lease_id", &TestWorkspaceLease::lease_id)
      .def("complete_for_test", &TestWorkspaceLease::complete_for_test);
  module.def(
      "reset_test_workspace_pool",
      &reset_test_workspace_pool,
      py::arg("capacity_bytes"));
  module.def(
      "acquire_test_lease", &acquire_test_lease, py::arg("size_bytes"));
}

}  // namespace ccdl_comm
