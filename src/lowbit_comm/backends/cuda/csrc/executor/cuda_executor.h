#pragma once

#include "compressed_work.h"

namespace ccdl_comm {

class CudaExecutor {
 public:
  std::shared_ptr<CompressedWork> run(
      py::object result,
      py::object transport_work = py::none(),
      py::object completion = py::none(),
      std::vector<py::object> resources = {},
      py::object callback = py::none()) const;
};

std::shared_ptr<CudaExecutor> create_cuda_executor();
void bind_cuda_executor(py::module_& module);

}  // namespace ccdl_comm
