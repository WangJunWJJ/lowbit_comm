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

 private:
  std::atomic<uint64_t> next_;
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
