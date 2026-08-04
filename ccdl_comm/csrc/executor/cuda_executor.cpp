#include "cuda_executor.h"

#include <utility>

namespace ccdl_comm {

std::shared_ptr<CompressedWork> CudaExecutor::run(
    py::object result,
    py::object transport_work,
    py::object completion,
    std::vector<py::object> resources,
    py::object callback) const {
  return std::make_shared<CompressedWork>(
      std::move(result),
      std::move(transport_work),
      std::move(completion),
      std::move(resources),
      std::move(callback));
}

std::shared_ptr<CudaExecutor> create_cuda_executor() {
  return std::make_shared<CudaExecutor>();
}

void bind_cuda_executor(py::module_& module) {
  py::class_<CudaExecutor, std::shared_ptr<CudaExecutor>>(module, "CudaExecutor")
      .def(
          "run",
          &CudaExecutor::run,
          py::arg("result"),
          py::arg("transport_work") = py::none(),
          py::arg("completion") = py::none(),
          py::arg("resources") = std::vector<py::object>{},
          py::arg("callback") = py::none());
  module.def("create_cuda_executor", &create_cuda_executor);
}

}  // namespace ccdl_comm
