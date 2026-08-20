#pragma once

#include "compressed_work.h"

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <atomic>
#include <cstdint>
#include <memory>

namespace ccdl_comm {

enum class ReducedShardCompression : uint8_t {
  kNative,
  kInt8,
};

enum class ReducedShardReduction : uint8_t {
  kSum,
  kMean,
};

class ReducedShardPlan {
 public:
  ReducedShardPlan(
      ReducedShardCompression compression,
      ReducedShardReduction reduction,
      at::ScalarType dtype,
      int64_t numel,
      int64_t logical_shard_length,
      int64_t rank,
      int64_t world_size,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group);

  std::shared_ptr<CudaWork> execute(torch::Tensor input);

 private:
  void validate_input(const torch::Tensor& input) const;
  std::shared_ptr<CudaWork> execute_native(torch::Tensor input);

  ReducedShardCompression compression_;
  ReducedShardReduction reduction_;
  at::ScalarType dtype_;
  int64_t numel_;
  int64_t logical_shard_length_;
  int64_t rank_;
  int64_t world_size_;
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
  uint64_t plan_id_;
  std::atomic<uint64_t> next_sequence_{1};
};

std::shared_ptr<ReducedShardPlan> create_reduced_shard_plan(
    py::dict config,
    py::object process_group);
void bind_reduced_shard_plan(py::module_& module);

}  // namespace ccdl_comm
