#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "quantization/quant_api.cuh"
#include "quantization/dequant_api.cuh"
#include "quantization/enum.cuh"


namespace py = pybind11;

PYBIND11_MODULE(ccdl_cuda_ops, m) {
    m.def("inplace_quantize", &inplace_quantize);
    m.def("quantize", &quantize);
    m.def("inplace_dequantize", &inplace_dequantize);
    m.def("dequantize", &dequantize);
    py::enum_<ReduceOP>(m, "ReduceOP")
        .value("SUM", ReduceOP::SUM)
        .value("NONE", ReduceOP::NONE)
        .value("MIN", ReduceOP::MIN)
        .value("MAX", ReduceOP::MAX)
        .export_values();
    py::enum_<QuantType>(m, "QuantType")
        .value("Linear", QuantType::Linear)
        .value("Normal", QuantType::Normal)
        .value("Uniform", QuantType::Uniform)
        .value("E3M0", QuantType::E3M0)
        .value("E2M1", QuantType::E2M1)
        .export_values();
    py::enum_<DType>(m, "DType")
        .value("FP16", DType::FP16)
        .value("BF16", DType::BF16)
        .value("FP32", DType::FP32)
        .export_values();
}