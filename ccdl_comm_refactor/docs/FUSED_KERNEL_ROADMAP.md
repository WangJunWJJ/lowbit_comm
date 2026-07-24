# CCDL fused kernel roadmap

This document separates the three optimization layers that are easy to mix up
when integrating CCDL into ParaScale.

## Current status

### 1. Payload-level fusion

Implemented by `make_fused_payload_all_gather`.

It packs the compressed buffer and tensor metadata, such as scales, into one
`uint8` all-gather payload. This reduces metadata collectives and helps larger
buckets, but it does not reduce the quantize/dequantize kernel chain itself.

### 2. CUDA compact codec layout

Implemented by `CompressionConfig(compact=True)`.

The original CUDA extension already contains compact quant/dequant kernels.
The refactored Python codec now exposes that path and forwards `compact=True`
to both extension calls. Benchmark scripts can compare:

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/collective_perf_compare.py \
  --output-json /tmp/ccdl_cuda_compact.json \
  --compact
```

Use `--no-compact` for the baseline layout.

### 3. AscendC fused quant/dequant kernel

Not implemented yet.

The safe CANN path currently uses ATen-on-NPU operations. A previous direct
ACLNN attempt was intentionally guarded behind `CCDL_COMM_EXPERIMENTAL_ACLNN`
because torch-npu op-plugin helpers introduced private ABI symbols that were
not exported by `libtorch_npu.so`.

The production Ascend path should be a dedicated AscendC custom op package,
not a direct dependency on torch-npu private C++ helpers.

## Next implementation target

The next durable breakthrough should be a true linear INT8 fused kernel:

1. One quant kernel computes per-group max-abs scale and writes quantized bytes
   plus compact fp16 scales.
2. One dequant kernel restores the reduced compressed payload directly to the
   requested dtype.
3. Both kernels expose stable Python extension symbols with safe import and
   torch fallback.
4. Benchmarks must compare native all-reduce, unfused CCDL, payload-fused CCDL,
   compact CUDA CCDL, and fused-kernel CCDL under the same bucket sizes.

Do not claim training throughput gains until the same model, data, batch size,
precision, hardware, warmup, and measurement window have been used.
