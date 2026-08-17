#pragma once

#include <cstdint>

namespace ccdl_comm {

struct LaunchToken final {
  uint64_t plan_id;
  uint64_t sequence;
};

}  // namespace ccdl_comm
