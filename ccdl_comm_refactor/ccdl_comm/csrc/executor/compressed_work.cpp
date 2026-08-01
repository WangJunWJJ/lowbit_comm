#include "compressed_work.h"

#include <utility>

namespace ccdl_comm {

CompressedWork::CompressedWork(
    py::object result,
    py::object transport_work,
    py::object completion,
    std::vector<py::object> resources,
    py::object callback)
    : result_(std::move(result)),
      transport_work_(std::move(transport_work)),
      completion_(std::move(completion)),
      resources_(std::move(resources)),
      callback_(std::move(callback)),
      callback_finished_(callback_.is_none()) {
  if (!transport_work_.is_none()) {
    try {
      native_transport_work_ = transport_work_.cast<c10::intrusive_ptr<c10d::Work>>();
    } catch (const py::cast_error&) {
      native_transport_work_.reset();
    }
    if (!native_transport_work_ && py::hasattr(transport_work_, "handle")) {
      try {
        native_transport_work_ = transport_work_.attr("handle")
                                     .cast<c10::intrusive_ptr<c10d::Work>>();
      } catch (const py::cast_error&) {
        native_transport_work_.reset();
      }
    }
  }
  if (!completion_.is_none() && THCPEvent_Check(completion_.ptr())) {
    native_completion_ = &reinterpret_cast<THCPEvent*>(completion_.ptr())->cuda_event;
  }
}

bool CompressedWork::query_object(const py::object& value) {
  if (value.is_none()) {
    return true;
  }
  if (py::hasattr(value, "is_completed")) {
    return value.attr("is_completed")().cast<bool>();
  }
  if (py::hasattr(value, "query")) {
    return value.attr("query")().cast<bool>();
  }
  return false;
}

bool CompressedWork::query() const {
  if (wait_state_.load(std::memory_order_acquire) == 2) {
    return true;
  }
  if (!query_transport()) {
    return false;
  }
  if (!callback_finished_) {
    return false;
  }
  return query_completion();
}

bool CompressedWork::query_transport() const {
  if (native_transport_work_) {
    py::gil_scoped_release release;
    return native_transport_work_->isCompleted();
  }
  return query_object(transport_work_);
}

bool CompressedWork::query_completion() const {
  if (native_completion_ != nullptr) {
    py::gil_scoped_release release;
    return native_completion_->query();
  }
  return query_object(completion_);
}

void CompressedWork::wait_transport() {
  if (native_transport_work_) {
    py::gil_scoped_release release;
    native_transport_work_->wait();
    return;
  }
  if (!transport_work_.is_none() && py::hasattr(transport_work_, "wait")) {
    transport_work_.attr("wait")();
  }
}

void CompressedWork::wait_completion() {
  if (native_completion_ != nullptr) {
    py::gil_scoped_release release;
    if (native_completion_->isCreated()) {
      native_completion_->block(
          at::cuda::getCurrentCUDAStream(native_completion_->device_index()));
    }
    return;
  }
  if (completion_.is_none()) {
    return;
  }
  if (py::hasattr(completion_, "wait")) {
    completion_.attr("wait")();
    return;
  }
  if (py::hasattr(completion_, "synchronize")) {
    completion_.attr("synchronize")();
  }
}

py::object CompressedWork::wait() {
  uint8_t expected = 0;
  bool execute_wait = wait_state_.compare_exchange_strong(
      expected,
      1,
      std::memory_order_acq_rel,
      std::memory_order_acquire);
  if (!execute_wait && expected != 2) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    py::gil_scoped_release release;
    state_cv_.wait(lock, [this] {
      return wait_state_.load(std::memory_order_acquire) == 2;
    });
  }

  if (execute_wait) {
    try {
      wait_transport();
      if (!callback_finished_) {
        callback_finished_ = true;
        result_ = callback_();
      }
      wait_completion();
    } catch (py::error_already_set& error) {
      error.restore();
      PyObject* error_type = nullptr;
      PyObject* error_value = nullptr;
      PyObject* error_traceback = nullptr;
      PyErr_Fetch(&error_type, &error_value, &error_traceback);
      error_type_ = py::reinterpret_steal<py::object>(error_type);
      error_value_ = py::reinterpret_steal<py::object>(error_value);
      error_traceback_ = error_traceback == nullptr
          ? py::none()
          : py::reinterpret_steal<py::object>(error_traceback);
      has_python_error_ = true;
    } catch (...) {
      cpp_error_ = std::current_exception();
    }
    resources_.clear();
    callback_ = py::none();
    wait_state_.store(2, std::memory_order_release);
    state_cv_.notify_all();
  }
  if (has_python_error_) {
    throw_cached_python_error();
  }
  if (cpp_error_) {
    std::rethrow_exception(cpp_error_);
  }
  return result_;
}

void CompressedWork::throw_cached_python_error() const {
  PyObject* error_type = error_type_.ptr();
  PyObject* error_value = error_value_.ptr();
  PyObject* error_traceback = error_traceback_.is_none() ? nullptr : error_traceback_.ptr();
  Py_XINCREF(error_type);
  Py_XINCREF(error_value);
  Py_XINCREF(error_traceback);
  PyErr_Restore(error_type, error_value, error_traceback);
  throw py::error_already_set();
}

py::object CompressedWork::get_future() const {
  if (!transport_work_.is_none() && py::hasattr(transport_work_, "get_future")) {
    return transport_work_.attr("get_future")();
  }
  return py::none();
}

py::object CompressedWork::result() {
  return wait();
}

py::tuple CompressedWork::resources() const {
  py::tuple result(resources_.size());
  for (size_t index = 0; index < resources_.size(); ++index) {
    result[index] = resources_[index];
  }
  return result;
}

bool CompressedWork::uses_native_transport() const {
  return static_cast<bool>(native_transport_work_);
}

bool CompressedWork::uses_native_completion() const {
  return native_completion_ != nullptr;
}

void bind_compressed_work(py::module_& module) {
  py::class_<CompressedWork, std::shared_ptr<CompressedWork>>(module, "CompressedWork")
      .def(
          py::init<
              py::object,
              py::object,
              py::object,
              std::vector<py::object>,
              py::object>(),
          py::arg("result"),
          py::arg("transport_work") = py::none(),
          py::arg("completion") = py::none(),
          py::arg("resources") = std::vector<py::object>{},
          py::arg("callback") = py::none())
      .def("wait", &CompressedWork::wait)
      .def("query", &CompressedWork::query)
      .def("get_future", &CompressedWork::get_future)
      .def("result", &CompressedWork::result)
      .def_property_readonly("resources", &CompressedWork::resources)
      .def_property_readonly(
          "uses_native_transport", &CompressedWork::uses_native_transport)
      .def_property_readonly(
          "uses_native_completion", &CompressedWork::uses_native_completion);
}

}  // namespace ccdl_comm
