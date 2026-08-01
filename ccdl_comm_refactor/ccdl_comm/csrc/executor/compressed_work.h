#pragma once

#include <torch/extension.h>
#include <torch/csrc/cuda/Event.h>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace py = pybind11;

namespace ccdl_comm {

class CompressedWork {
 public:
  CompressedWork(
      py::object result,
      py::object transport_work = py::none(),
      py::object completion = py::none(),
      std::vector<py::object> resources = {},
      py::object callback = py::none());

  bool query() const;
  py::object wait();
  py::object get_future() const;
  py::object result();
  py::tuple resources() const;
  bool uses_native_transport() const;
  bool uses_native_completion() const;

 private:
  static bool query_object(const py::object& value);
  bool query_transport() const;
  bool query_completion() const;
  void wait_transport();
  void wait_completion();
  [[noreturn]] void throw_cached_python_error() const;

  py::object result_;
  py::object transport_work_;
  py::object completion_;
  std::vector<py::object> resources_;
  py::object callback_;
  c10::intrusive_ptr<c10d::Work> native_transport_work_;
  at::cuda::CUDAEvent* native_completion_{nullptr};
  std::atomic<uint8_t> wait_state_{0};
  bool callback_finished_{false};
  bool has_python_error_{false};
  py::object error_type_{py::none()};
  py::object error_value_{py::none()};
  py::object error_traceback_{py::none()};
  std::exception_ptr cpp_error_;
  std::mutex state_mutex_;
  std::condition_variable state_cv_;
};

void bind_compressed_work(py::module_& module);

}  // namespace ccdl_comm
