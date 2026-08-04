# CCDL CANN Kernel Design

## Goal

Implement an Ascend CANN backend for CCDL low-bit communication so ParaScale can run real compressed communication on Ascend/HCCL instead of the current portable torch fallback.

## Scope

The implementation targets a complete CANN backend in incremental, verifiable slices:

1. Build and safe-import infrastructure for an optional `ccdl_cann_ops` extension.
2. AscendC kernels for group-wise linear INT8 quantization and dequantization.
3. Python codec wrappers returning the existing `CompressedPayload` contract.
4. HCCL benchmark and DDP hook validation using the CANN codec.
5. Communication format optimization by packing quantized values and scale metadata into fewer HCCL payloads after the baseline CANN codec is correct.
6. Extension points for 4-bit and non-linear quantization modes after INT8 is validated.

## Non-Goals

- Do not replace HCCL or implement a new collective backend.
- Do not change the public CUDA extension module name or behavior.
- Do not make CANN mandatory for importing `ccdl_comm`.
- Do not claim performance wins until verified on the 8-card Ascend server.

## Architecture

The package will add an Ascend-specific backend alongside the existing CUDA backend:

```text
ccdl_comm/
  ascend/
    __init__.py
    loader.py
    codec.py
  build/
    cann.py
  csrc_ascend/
    pybind.cpp
    quant_linear_int8.cpp
    kernels/
      quant_linear_int8.cpp
      dequant_linear_int8.cpp
```

`ccdl_comm.ascend.loader.load_cann_extension()` mirrors the CUDA loader: it catches missing extension, missing CANN runtime, and import failures and returns structured status instead of raising during ParaScale planning.

`ccdl_comm.ascend.codec.quantize_tensor_cann()` and `dequantize_tensor_cann()` match the fallback codec signatures:

```python
quantize_tensor_cann(tensor, config, *, extension_status=None) -> CompressedPayload
dequantize_tensor_cann(payload, shape, config, dtype, *, extension_status=None) -> Tensor
```

This keeps `compressed_all_reduce`, `compressed_all_gather`, and `create_ddp_comm_hook` unchanged except for choosing which codec to inject.

## Kernel Semantics

Initial CANN kernels implement linear group-wise INT8:

```text
groups = reshape(pad(flat(input), group_size), [-1, group_size])
scale[group] = max(abs(groups[group])) / 127
q[group, i] = round(clamp(groups[group, i] / scale[group], -127, 127))
restored[group, i] = q[group, i] * scale[group]
```

Supported initial configuration:

- `bit=8`
- `quant_type="linear"`
- `group_size in {16, 32, 64}`
- `topk=0`
- source dtype: `fp16`, `bf16`, `fp32`

Unsupported modes must fail with clear `UnsupportedCollective` or `CCDLUnavailableError` style messages rather than silently falling back inside a performance benchmark.

## Communication Format

Baseline CANN payload keeps the current correctness-preserving format:

```text
buffer: int8 tensor, flattened padded quantized values
metadata:
  scales: fp32 tensor, one scale per group
  original_numel: Python int
```

After the baseline CANN codec is verified, add a packed payload format:

```text
payload_tensor = [header, packed_scales, int8_values]
```

The first optimization target is reducing HCCL all-gather calls from two tensor gathers (`buffer`, `scales`) to one tensor gather. Header fields must be fixed-width integer values so every rank can parse the payload without CPU synchronization.

## Build Strategy

CANN build support must be opt-in:

```text
CCDL_COMM_BUILD_CANN=1
```

Import safety remains mandatory:

- missing torch-npu: unavailable status
- missing CANN toolkit: unavailable status
- missing compiled `ccdl_cann_ops`: unavailable status
- broken ABI/import error: unavailable status with reason

The Ascend server has CANN 9.0.0, `opbuild`, `msopgen`, AscendC headers, and `torch_npu.utils.cpp_extension`; the first implementation should prefer `torch_npu.utils.cpp_extension` if it can compile a small extension, and fall back to explicit CANN toolchain commands only if needed.

## Testing and Validation

Local tests:

- loader safe import
- setuptools/build kwargs for optional CANN
- codec rejects unsupported config
- codec wrapper produces `CompressedPayload`
- benchmark script includes CANN path

Ascend remote tests:

- CANN extension build smoke
- single-rank quant/dequant relative L2 `< 0.02`
- two-rank HCCL benchmark vs torch fallback and native all-reduce
- NPU DDP hook smoke using CANN codec

Performance reports must include:

- backend and device type
- world size
- tensor shape/numel
- dtype, bit, group size
- warmup/repeat
- native HCCL latency
- CANN compressed latency
- relative L2
- torch/torch-npu/CANN versions where available

## Risks

- AscendC custom op examples may differ across CANN versions.
- Dynamic extension compilation may require environment variables not present in the current image.
- A fully fused quant/dequant path still cannot fuse the HCCL collective itself; the fusion boundary is codec-side.
- 4-bit packing and non-linear quantization are separate correctness/performance tasks and should follow after INT8 CANN validation.
