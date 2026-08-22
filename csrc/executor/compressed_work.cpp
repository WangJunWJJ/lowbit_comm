#include "compressed_work.h"

#include <ATen/cuda/CUDAContext.h>

#include <exception>
#include <thread>
#include <utility>

namespace ccdl_comm {

namespace {

template <typename Callback>
class NoexceptScopeExit final {
 public:
  explicit NoexceptScopeExit(Callback callback)
      : callback_(std::move(callback)) {}

  ~NoexceptScopeExit() noexcept {
    callback_();
  }

  NoexceptScopeExit(const NoexceptScopeExit&) = delete;
  NoexceptScopeExit& operator=(const NoexceptScopeExit&) = delete;

 private:
  Callback callback_;
};

template <typename Callback>
NoexceptScopeExit<Callback> make_noexcept_scope_exit(Callback callback) {
  return NoexceptScopeExit<Callback>(std::move(callback));
}

}  // namespace

CudaWork::CudaWork(
    py::object value,
    LaunchToken token,
    TestEventState test_state)
    : value_(std::move(value)),
      token_(token),
      test_state_(test_state) {
  if (test_state_ == TestEventState::kNative) {
    event_.record(at::cuda::getCurrentCUDAStream());
  } else if (test_state_ == TestEventState::kSuccess) {
    state_.store(WorkState::kSucceeded, std::memory_order_release);
  } else if (test_state_ == TestEventState::kFailure) {
    failure_message_ = "test CUDA failure";
    state_.store(WorkState::kFailed, std::memory_order_release);
  }
}

CudaWork::CudaWork(
    py::object value,
    LaunchToken token,
    std::unique_ptr<WorkspaceLease>& lease,
    CudaEventFailureInjection event_failure)
    : value_(std::move(value)),
      token_(token),
      test_state_(TestEventState::kNative),
      event_failure_(event_failure) {
  try {
    if (event_failure_ == CudaEventFailureInjection::kRecord) {
      throw CudaExecutionError("test CUDA event record failure");
    }
    event_.record(at::cuda::getCurrentCUDAStream());
  } catch (...) {
    if (lease) {
      lease->quarantine();
    }
    throw;
  }
  lease_ = std::move(lease);
}

CudaWork::~CudaWork() {
  bool completion_proven =
      state_.load(std::memory_order_acquire) != WorkState::kPending;
  try {
    if (test_state_ == TestEventState::kNative &&
        !completion_proven) {
      synchronize_event();
      completion_proven = true;
    }
  } catch (...) {
    if (lease_) {
      lease_->quarantine();
    }
  }
  if (!completion_proven && lease_) {
    lease_->quarantine();
  }
  lease_.reset();
}

bool CudaWork::event_ready() const {
  if (test_state_ == TestEventState::kPending) {
    return false;
  }
  if (test_state_ != TestEventState::kNative) {
    return true;
  }
  return event_.query();
}

void CudaWork::synchronize_event() const {
  if (test_state_ == TestEventState::kNative) {
    synchronize_count_.fetch_add(1, std::memory_order_relaxed);
    if (event_failure_ == CudaEventFailureInjection::kSynchronize) {
      throw CudaExecutionError("test CUDA event synchronize failure");
    }
    event_.synchronize();
  }
}

bool CudaWork::is_completed() const {
  if (state_.load(std::memory_order_acquire) != WorkState::kPending) {
    return true;
  }
  return event_ready();
}

bool CudaWork::terminal_published() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return terminal_published_;
}

void CudaWork::finish_once() {
  WaitPhase expected = WaitPhase::kNotStarted;
  const bool owns_wait = wait_phase_.compare_exchange_strong(
      expected,
      WaitPhase::kRunning,
      std::memory_order_acq_rel,
      std::memory_order_acquire);
  if (!owns_wait) {
    if (expected != WaitPhase::kFinished) {
      std::unique_lock<std::mutex> lock(mutex_);
      py::gil_scoped_release release;
      while (wait_phase_.load(std::memory_order_acquire) !=
             WaitPhase::kFinished) {
        if (!latch_before_condition_wait_for_test(lock)) {
          continue;
        }
        condition_.wait(lock);
      }
    }
    return;
  }

  bool finalized = false;
  auto finalizer = make_noexcept_scope_exit([this, &finalized]() noexcept {
    if (!finalized) {
      finalize_wait_owner_noexcept();
    }
  });
  try {
    {
      py::gil_scoped_release release;
      synchronize_event();
    }
    if (test_state_ == TestEventState::kFailure) {
      state_.store(WorkState::kFailed, std::memory_order_release);
    } else {
      state_.store(WorkState::kSucceeded, std::memory_order_release);
    }
  } catch (const std::exception& error) {
    try {
      failure_message_ = error.what();
    } catch (...) {
      fallback_failure_message_ =
          "CUDA completion failure message allocation failed";
    }
    state_.store(WorkState::kFailed, std::memory_order_release);
  } catch (...) {
    fallback_failure_message_ = "unknown CUDA completion failure";
    state_.store(WorkState::kFailed, std::memory_order_release);
  }
  if (wait_latch_enabled_.load(std::memory_order_acquire)) {
    owner_completion_blocked_.store(true, std::memory_order_release);
    py::gil_scoped_release release;
    while (!allow_completion_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    finalize_wait_owner_noexcept();
    finalized = true;
  }
}

void CudaWork::finalize_wait_owner_noexcept() noexcept {
  WorkState state = state_.load(std::memory_order_acquire);
  if (state == WorkState::kPending) {
    fallback_failure_message_ =
        "CUDA completion owner exited without terminal state";
    state_.store(WorkState::kFailed, std::memory_order_release);
    state = WorkState::kFailed;
  }
  if (state == WorkState::kFailed && lease_) {
    lease_->quarantine();
  }

  terminal_publish_attempted_.store(true, std::memory_order_release);
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    wait_phase_.store(WaitPhase::kFinished, std::memory_order_release);
    terminal_published_ = true;
  } catch (...) {
    std::terminate();
  }
  condition_.notify_all();
  lease_.reset();
}

bool CudaWork::latch_before_condition_wait_for_test(
    std::unique_lock<std::mutex>& lock) noexcept {
  if (!wait_latch_enabled_.load(std::memory_order_acquire)) {
    return true;
  }
  const uint64_t arrived =
      losers_arrived_.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (arrived < expected_losers_.load(std::memory_order_acquire)) {
    lock.unlock();
    while (!release_losers_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    lock.lock();
    return false;
  }
  while (!release_losers_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  return true;
}

[[noreturn]] void CudaWork::throw_failure() const {
  if (!failure_message_.empty()) {
    throw CudaExecutionError(failure_message_);
  }
  throw CudaExecutionError(
      fallback_failure_message_ != nullptr
      ? fallback_failure_message_
      : "CUDA completion failed");
}

py::object CudaWork::wait() {
  if (test_state_ == TestEventState::kPending) {
    throw CudaExecutionError("CUDA work is not completed");
  }
  finish_once();
  if (state_.load(std::memory_order_acquire) == WorkState::kFailed) {
    throw_failure();
  }
  return value_;
}

py::object CudaWork::result() {
  if (!is_completed()) {
    throw CudaExecutionError("CUDA work is not completed");
  }
  return wait();
}

LaunchToken CudaWork::launch_token() const {
  return token_;
}

uint64_t CudaWork::synchronize_count_for_test() const noexcept {
  return synchronize_count_.load(std::memory_order_relaxed);
}

void CudaWork::enable_wait_latch_for_test(uint64_t expected_losers) {
  if (expected_losers == 0) {
    throw py::value_error("expected_losers must be positive");
  }
  expected_losers_.store(expected_losers, std::memory_order_release);
  wait_latch_enabled_.store(true, std::memory_order_release);
}

py::dict CudaWork::wait_latch_state_for_test() const {
  py::dict state;
  state["owner_completion_blocked"] =
      owner_completion_blocked_.load(std::memory_order_acquire);
  state["losers_arrived"] =
      losers_arrived_.load(std::memory_order_acquire);
  state["terminal_publish_attempted"] =
      terminal_publish_attempted_.load(std::memory_order_acquire);
  return state;
}

void CudaWork::allow_completion_for_test() noexcept {
  allow_completion_.store(true, std::memory_order_release);
}

void CudaWork::release_losers_for_test() noexcept {
  release_losers_.store(true, std::memory_order_release);
}

namespace {

class TestMonotonicAllocator final {
 public:
  explicit TestMonotonicAllocator(uint64_t initial) : allocator_(initial) {}

  uint64_t allocate() {
    return allocator_.allocate("CUDA plan identity space is exhausted");
  }

 private:
  SaturatingMonotonicAllocator allocator_;
};

std::shared_ptr<CudaWork> make_test_work(
    const std::string& event_state,
    py::object value) {
  TestEventState state;
  if (event_state == "pending") {
    state = TestEventState::kPending;
  } else if (event_state == "success") {
    state = TestEventState::kSuccess;
  } else if (event_state == "failure") {
    state = TestEventState::kFailure;
  } else {
    throw py::value_error("event_state must be pending, success, or failure");
  }
  return std::make_shared<CudaWork>(
      std::move(value), LaunchToken{0, 0}, state);
}

}  // namespace

CudaEventFailureInjection parse_cuda_event_failure_injection_for_test(
    const std::string& value) {
  if (value == "none") {
    return CudaEventFailureInjection::kNone;
  }
  if (value == "record") {
    return CudaEventFailureInjection::kRecord;
  }
  if (value == "synchronize") {
    return CudaEventFailureInjection::kSynchronize;
  }
  throw py::value_error(
      "event failure must be none, record, or synchronize");
}

void bind_cuda_work(py::module_& module) {
  py::register_exception<CudaExecutionError>(
      module,
      "_CudaExecutionError",
      py::module_::import("lowbit_comm").attr("ExecutionError").ptr());
  py::class_<LaunchToken>(module, "LaunchToken")
      .def_readonly("plan_id", &LaunchToken::plan_id)
      .def_readonly("sequence", &LaunchToken::sequence);
  py::class_<TestMonotonicAllocator>(module, "_TestMonotonicAllocator")
      .def("allocate", &TestMonotonicAllocator::allocate);
  py::class_<CudaWork, std::shared_ptr<CudaWork>>(
      module, "CudaWork", py::dynamic_attr())
      .def("is_completed", &CudaWork::is_completed)
      .def("_is_terminal_for_feedback", &CudaWork::terminal_published)
      .def("wait", &CudaWork::wait)
      .def("result", &CudaWork::result)
      .def(
          "_synchronize_count_for_test",
          &CudaWork::synchronize_count_for_test)
      .def(
          "_enable_wait_latch_for_test",
          &CudaWork::enable_wait_latch_for_test)
      .def(
          "_wait_latch_state_for_test",
          &CudaWork::wait_latch_state_for_test)
      .def(
          "_allow_completion_for_test",
          &CudaWork::allow_completion_for_test)
      .def(
          "_release_losers_for_test",
          &CudaWork::release_losers_for_test)
      .def("launch_token", &CudaWork::launch_token);
  module.def(
      "make_test_work",
      &make_test_work,
      py::arg("event_state"),
      py::arg("value"));
  module.def(
      "make_test_monotonic_allocator",
      [](uint64_t initial) {
        return std::make_unique<TestMonotonicAllocator>(initial);
      },
      py::arg("initial"));
}

}  // namespace ccdl_comm
