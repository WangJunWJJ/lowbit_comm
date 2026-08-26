#include "workspace_pool.h"

#include "../executor/compressed_work.h"

#include <utility>

namespace ccdl_comm {

WorkspaceLease::WorkspaceLease(
    std::shared_ptr<WorkspacePool> pool,
    uint64_t lease_id,
    size_t size_bytes,
    torch::Tensor storage,
    std::unique_ptr<WorkspaceQuarantineNode> quarantine_node)
    : pool_(std::move(pool)),
      lease_id_(lease_id),
      size_bytes_(size_bytes),
      storage_(std::move(storage)),
      quarantine_node_(std::move(quarantine_node)) {}

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
  if (terminal_) {
    return;
  }
  terminal_ = true;
  quarantine_node_.reset();
  pool_->release(size_bytes_, std::move(storage_));
}

void WorkspaceLease::quarantine() noexcept {
  if (terminal_) {
    return;
  }
  terminal_ = true;
  try {
    quarantine_node_->storage = std::move(storage_);
    pool_->quarantine(std::move(quarantine_node_), pool_);
  } catch (...) {
    quarantine_node_.release();
  }
}

WorkspacePool::WorkspacePool(size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes) {}

std::unique_ptr<WorkspaceLease> WorkspacePool::acquire(size_t size_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (poisoned_.load(std::memory_order_acquire)) {
    throw CudaExecutionError("workspace pool is quarantined");
  }
  if (size_bytes > capacity_bytes_ - used_bytes_) {
    throw CudaExecutionError("workspace pool capacity exhausted");
  }
  auto quarantine_node =
      std::make_unique<WorkspaceQuarantineNode>();
  torch::Tensor storage;
  auto cached = free_buffers_.find(size_bytes);
  if (cached != free_buffers_.end() && !cached->second.empty()) {
    storage = std::move(cached->second.back());
    cached->second.pop_back();
    cached_bytes_ -= size_bytes;
    if (cached->second.empty()) {
      free_buffers_.erase(cached);
    }
  } else {
    while (used_bytes_ + cached_bytes_ + size_bytes > capacity_bytes_ &&
           !free_buffers_.empty()) {
      auto victim = free_buffers_.begin();
      cached_bytes_ -= victim->first * victim->second.size();
      free_buffers_.erase(victim);
    }
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA);
    storage = torch::empty(
        {static_cast<int64_t>(size_bytes)}, options);
    allocation_count_ += 1;
  }
  const uint64_t lease_id = next_lease_id_++;
  used_bytes_ += size_bytes;
  return std::make_unique<WorkspaceLease>(
      shared_from_this(),
      lease_id,
      size_bytes,
      std::move(storage),
      std::move(quarantine_node));
}

void WorkspacePool::release(size_t size_bytes, torch::Tensor storage) {
  std::lock_guard<std::mutex> lock(mutex_);
  used_bytes_ -= size_bytes;
  if (!poisoned_.load(std::memory_order_acquire)) {
    free_buffers_[size_bytes].push_back(std::move(storage));
    cached_bytes_ += size_bytes;
  }
}

void WorkspacePool::quarantine(
    std::unique_ptr<WorkspaceQuarantineNode> node,
    const std::shared_ptr<WorkspacePool>& self) noexcept {
  poisoned_.store(true, std::memory_order_release);
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    node->next = std::move(quarantine_head_);
    quarantine_head_ = std::move(node);
    quarantine_self_ = self;
  } catch (...) {
    node.release();
  }
}

uint64_t WorkspacePool::allocation_count_for_test() {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocation_count_;
}

TestWorkspaceLease::TestWorkspaceLease(
    std::unique_ptr<WorkspaceLease> lease)
    : lease_id_(lease->lease_id()), lease_(std::move(lease)) {}

uint64_t TestWorkspaceLease::lease_id() const {
  return lease_id_;
}

uint64_t TestWorkspaceLease::storage_data_ptr() const {
  if (!lease_) {
    throw CudaExecutionError("test workspace lease is already complete");
  }
  return reinterpret_cast<uint64_t>(lease_->storage().data_ptr());
}

void TestWorkspaceLease::complete_for_test() {
  lease_.reset();
}

void TestWorkspaceLease::quarantine_for_test() {
  lease_->quarantine();
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

uint64_t test_workspace_pool_allocation_count() {
  if (!test_pool) {
    throw CudaExecutionError("workspace pool is not initialized");
  }
  return test_pool->allocation_count_for_test();
}

}  // namespace

void bind_workspace_pool(py::module_& module) {
  py::class_<TestWorkspaceLease, std::shared_ptr<TestWorkspaceLease>>(
      module, "_TestWorkspaceLease")
      .def_property_readonly("lease_id", &TestWorkspaceLease::lease_id)
      .def_property_readonly(
          "storage_data_ptr", &TestWorkspaceLease::storage_data_ptr)
      .def("complete_for_test", &TestWorkspaceLease::complete_for_test)
      .def(
          "quarantine_for_test",
          &TestWorkspaceLease::quarantine_for_test);
  module.def(
      "reset_test_workspace_pool",
      &reset_test_workspace_pool,
      py::arg("capacity_bytes"));
  module.def(
      "acquire_test_lease", &acquire_test_lease, py::arg("size_bytes"));
  module.def(
      "test_workspace_pool_allocation_count",
      &test_workspace_pool_allocation_count);
}

}  // namespace ccdl_comm
