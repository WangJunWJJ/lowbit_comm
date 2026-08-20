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

inline uint64_t allocate_cuda_plan_id() {
  static std::atomic<uint64_t> next_plan_id{1};
  const uint64_t plan_id =
      next_plan_id.fetch_add(1, std::memory_order_relaxed);
  if (plan_id == 0 || plan_id == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("CUDA plan identity space is exhausted");
  }
  return plan_id;
}

}  // namespace ccdl_comm
