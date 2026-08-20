#pragma once

#include "../runtime/launch_token.h"
#include "../runtime/workspace_pool.h"

#include <ATen/cuda/CUDAEvent.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace ccdl_comm {

class CudaExecutionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class TestEventState : uint8_t {
  kNative,
  kPending,
  kSuccess,
  kFailure,
};

enum class WorkState : uint8_t {
  kPending,
  kSucceeded,
  kFailed,
};

enum class WaitPhase : uint8_t {
  kNotStarted,
  kRunning,
  kFinished,
};

enum class CudaEventFailureInjection : uint8_t {
  kNone,
  kRecord,
  kSynchronize,
};

class CudaWork {
 public:
  CudaWork(
      py::object value,
      LaunchToken token,
      TestEventState test_state = TestEventState::kNative);
  CudaWork(
      py::object value,
      LaunchToken token,
      std::unique_ptr<WorkspaceLease>& lease,
      CudaEventFailureInjection event_failure =
          CudaEventFailureInjection::kNone);
  ~CudaWork();

  CudaWork(const CudaWork&) = delete;
  CudaWork& operator=(const CudaWork&) = delete;

  bool is_completed() const;
  py::object wait();
  py::object result();
  LaunchToken launch_token() const;
  uint64_t synchronize_count_for_test() const noexcept;
  void enable_wait_latch_for_test(uint64_t expected_losers);
  py::dict wait_latch_state_for_test() const;
  void allow_completion_for_test() noexcept;
  void release_losers_for_test() noexcept;

 private:
  bool event_ready() const;
  void synchronize_event() const;
  void finish_once();
  void finalize_wait_owner_noexcept() noexcept;
  bool latch_before_condition_wait_for_test(
      std::unique_lock<std::mutex>& lock) noexcept;
  [[noreturn]] void throw_failure() const;

  py::object value_;
  LaunchToken token_;
  std::unique_ptr<WorkspaceLease> lease_;
  TestEventState test_state_;
  CudaEventFailureInjection event_failure_{
      CudaEventFailureInjection::kNone};
  at::cuda::CUDAEvent event_;
  std::atomic<WorkState> state_{WorkState::kPending};
  std::atomic<WaitPhase> wait_phase_{WaitPhase::kNotStarted};
  std::string failure_message_;
  const char* fallback_failure_message_{nullptr};
  mutable std::atomic<uint64_t> synchronize_count_{0};
  std::atomic<bool> wait_latch_enabled_{false};
  mutable std::atomic<bool> owner_completion_blocked_{false};
  std::atomic<bool> allow_completion_{false};
  std::atomic<uint64_t> expected_losers_{0};
  std::atomic<uint64_t> losers_arrived_{0};
  std::atomic<bool> terminal_publish_attempted_{false};
  std::atomic<bool> release_losers_{false};
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

CudaEventFailureInjection parse_cuda_event_failure_injection_for_test(
    const std::string& value);

void bind_cuda_work(py::module_& module);

}  // namespace ccdl_comm
