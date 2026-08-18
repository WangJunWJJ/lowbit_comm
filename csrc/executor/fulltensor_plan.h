#pragma once

#include "compressed_work.h"

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace ccdl_comm {

enum class FullTensorCompression : uint8_t {
  kNative,
  kInt8,
};

enum class FullTensorReduction : uint8_t {
  kSum,
  kMean,
};

class FullTensorPlan {
 public:
  FullTensorPlan(
      FullTensorCompression compression,
      FullTensorReduction reduction,
      at::ScalarType dtype,
      int64_t numel,
      int64_t rank,
      int64_t world_size,
      int64_t group_size,
      int64_t payload_bytes_per_rank,
      int64_t workspace_bytes,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group);

  std::shared_ptr<CudaWork> execute(torch::Tensor input);

 private:
  std::shared_ptr<CudaWork> execute_native(torch::Tensor input);
  std::shared_ptr<CudaWork> execute_int8(torch::Tensor input);
  void validate_input(const torch::Tensor& input) const;

  FullTensorCompression compression_;
  FullTensorReduction reduction_;
  at::ScalarType dtype_;
  int64_t numel_;
  int64_t rank_;
  int64_t world_size_;
  int64_t group_size_;
  int64_t payload_bytes_per_rank_;
  int64_t workspace_bytes_;
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
  std::shared_ptr<WorkspacePool> workspace_pool_;
  uint64_t plan_id_;
  std::atomic<uint64_t> next_sequence_{1};
};

std::shared_ptr<FullTensorPlan> create_fulltensor_plan(
    py::dict config,
    py::object process_group);
void bind_fulltensor_plan(py::module_& module);

}  // namespace ccdl_comm
