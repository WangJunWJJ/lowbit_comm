#include "compressed_work.h"

#include <ATen/cuda/CUDAContext.h>

#include <utility>

namespace ccdl_comm {

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
      condition_.wait(lock, [this] {
        return wait_phase_.load(std::memory_order_acquire) ==
            WaitPhase::kFinished;
      });
    }
    return;
  }

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
    failure_message_ = error.what();
    state_.store(WorkState::kFailed, std::memory_order_release);
    if (lease_) {
      lease_->quarantine();
    }
  } catch (...) {
    failure_message_ = "unknown CUDA completion failure";
    state_.store(WorkState::kFailed, std::memory_order_release);
    if (lease_) {
      lease_->quarantine();
    }
  }
  lease_.reset();
  wait_phase_.store(WaitPhase::kFinished, std::memory_order_release);
  condition_.notify_all();
}

[[noreturn]] void CudaWork::throw_failure() const {
  throw CudaExecutionError(failure_message_);
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
      .def("wait", &CudaWork::wait)
      .def("result", &CudaWork::result)
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
