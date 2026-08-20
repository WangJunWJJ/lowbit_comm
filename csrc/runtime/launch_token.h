#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ccdl_comm {

struct LaunchToken final {
  uint64_t plan_id;
  uint64_t sequence;
};

class SaturatingMonotonicAllocator final {
 public:
  explicit SaturatingMonotonicAllocator(uint64_t initial = 1)
      : next_(initial) {}

  uint64_t allocate(const char* exhaustion_message) {
    uint64_t candidate = next_.load(std::memory_order_relaxed);
    while (candidate != std::numeric_limits<uint64_t>::max()) {
      if (next_.compare_exchange_weak(
              candidate,
              candidate + 1,
              std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return candidate;
      }
    }
    throw std::overflow_error(exhaustion_message);
  }

  void exhaust_for_test() {
    next_.store(
        std::numeric_limits<uint64_t>::max(),
        std::memory_order_release);
  }

 private:
  std::atomic<uint64_t> next_;
};

class LaunchSideEffectCounters final {
 public:
  void mark_allocation() {
    allocation_.fetch_add(1, std::memory_order_relaxed);
  }
  void mark_workspace_acquire() {
    workspace_acquire_.fetch_add(1, std::memory_order_relaxed);
  }
  void mark_transport_launch() {
    transport_launch_.fetch_add(1, std::memory_order_relaxed);
  }
  void mark_kernel_launch() {
    kernel_launch_.fetch_add(1, std::memory_order_relaxed);
  }
  void mark_work_publish() {
    work_publish_.fetch_add(1, std::memory_order_relaxed);
  }

  uint64_t allocation() const {
    return allocation_.load(std::memory_order_relaxed);
  }
  uint64_t workspace_acquire() const {
    return workspace_acquire_.load(std::memory_order_relaxed);
  }
  uint64_t transport_launch() const {
    return transport_launch_.load(std::memory_order_relaxed);
  }
  uint64_t kernel_launch() const {
    return kernel_launch_.load(std::memory_order_relaxed);
  }
  uint64_t work_publish() const {
    return work_publish_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t> allocation_{0};
  std::atomic<uint64_t> workspace_acquire_{0};
  std::atomic<uint64_t> transport_launch_{0};
  std::atomic<uint64_t> kernel_launch_{0};
  std::atomic<uint64_t> work_publish_{0};
};

inline uint64_t allocate_cuda_plan_id() {
  static SaturatingMonotonicAllocator next_plan_id{1};
  return next_plan_id.allocate("CUDA plan identity space is exhausted");
}

inline uint64_t allocate_cuda_sequence(
    SaturatingMonotonicAllocator& next_sequence) {
  return next_sequence.allocate("CUDA launch sequence space is exhausted");
}

}  // namespace ccdl_comm
