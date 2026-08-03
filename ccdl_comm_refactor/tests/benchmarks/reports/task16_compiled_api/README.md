# Task 16 Compiled P2P and Dynamic All-Gather Gate

## Scope

Task 16 moves the existing quantized point-to-point and dynamic all-gather
paths behind reusable CUDA executors.  It does not replace the validated codec
or PyTorch/NCCL transports.

The compiled P2P endpoint freezes direction, peer, process group, tag, tensor
shape, dtype, compression config, and payload size.  Every in-flight Work owns
the source/output tensor, quantized payload, and protocol metadata until
completion.  Transport failures remain deferred to and stable across
`wait()` calls.

The compiled dynamic all-gather endpoint freezes a common cross-rank shape
class.  Runtime metadata uses protocol version 1 and carries only the actual
shape, dtype, and payload length.  Communication uses the fixed class capacity
and trims each payload before dequantization, so padding cannot leak into the
result.  Zero-length tensors bypass the native quantization kernel.

## Environment

- Candidate base commit: `3814585`
- Host: `user@192.168.8.156 -p 360`
- GPU: 2 x NVIDIA RTX A6000
- Container: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA runtime: `12.6`
- Compression: linear INT8, group size 64
- CUDA extension: `sm_86`, SHA256
  `bcd3db260aec9d3116576b64e4dbdd21b2c5f8852e7580a16ab3a37834bc5904`

## Commands

```bash
torchrun --standalone --nproc-per-node=2 \
  tests/distributed/point_to_point_smoke.py \
  --numel=1048576 --dtype=fp16 --bit=8 --group-size=64 \
  --output-json=tests/benchmarks/reports/task16_compiled_api/raw/2gpu_p2p.json

torchrun --standalone --nproc-per-node=2 \
  tests/distributed/dynamic_all_gather_smoke.py \
  --base-numel=524288 --dtype=fp16 --bit=8 --group-size=64 \
  --output-json=tests/benchmarks/reports/task16_compiled_api/raw/2gpu_dynamic_gather.json

torchrun --standalone --nproc-per-node=4 \
  tests/distributed/dynamic_all_gather_smoke.py \
  --base-numel=524288 --dtype=fp16 --bit=8 --group-size=64 \
  --output-json=tests/benchmarks/reports/task16_compiled_api/raw/4gpu_dynamic_gather.json
```

## Results

| Path | Check | Result |
|---|---|---:|
| compiled blocking P2P | relative L2 | 0.005938 |
| compiled asynchronous P2P | relative L2 | 0.005938 |
| compiled dynamic gather | maximum relative L2 | 0.005954 |
| dynamic boundary shapes | `(0, 63, 64, 65)` | exact shapes |
| dynamic boundary values | maximum relative L2 | 0.003950 |
| 4-GPU dynamic boundary values | maximum relative L2 | 0.003918 |
| metadata protocol | version | 1 |
| shape-class cache | entries for large/boundary classes | 2 |

Both distributed commands exited zero.  The local full suite completed with
`681 passed, 30 skipped`; focused container tests completed with `17 passed`
on both A6000 nodes.

## Constraints

- Every rank must compile dynamic all-gather with the same shape-class bound.
- Task 16 removes repeated control-plane parsing but does not claim a kernel or
  transport speedup.  Latency remains dominated by quantization and NCCL/P2P.
- Dynamic shape metadata still uses `all_gather_object`; replacing it with a
  device-resident fixed metadata packet is a later data-plane optimization.
