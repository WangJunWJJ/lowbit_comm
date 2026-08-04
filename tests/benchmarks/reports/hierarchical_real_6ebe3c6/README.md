# A6000 real hierarchical transport prototype validation, 2026-07-28

## Scope

This report validates the first real torch.distributed-backed hierarchical
compressed transport prototype. The implementation constructs local 2-rank
groups on a 4-GPU run, performs compressed local gather/reduce, runs a native
leader all-reduce over restored partial tensors, then broadcasts the full result
inside each local group.

Performance is the first priority. Because this prototype is slower than the
validated all-gather path, `strategy="auto"` keeps falling back to all-gather
even when the hierarchical transport is available. The hierarchical transport
remains explicit opt-in for further profiling.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- GPUs used: 4, selected from GPU `0,1,2,3`
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Script: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`

## Workload

- Model: synthetic dense MLP
- Parameters: 46,137,344
- Steps: 50
- Warmup steps excluded from timing: 10
- Per-rank batch size: 8
- Input dim: 2048
- Width: 4096
- Depth: 3
- Output dim: 1024
- Dtype: FP32
- Optimizer: SGD, lr `1e-4`
- CCDL mode: INT8, group size 64, error feedback enabled,
  `async_gather=true`, `async_error_feedback=true`
- Hierarchical local group size: 2

## Result

| GPUs | Requested strategy | Hierarchical transport | Selected strategy | Requires fallback | Samples/s | Ratio vs all-gather | Avg step ms | Peak memory MB | Loss |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | all_gather | false | all_gather | false | 1092.71 | 1.000 | 29.285 | 1494.16 | 0.9986362767 |
| 4 | hierarchical | true | hierarchical | false | 499.79 | 0.457 | 64.028 | 922.91 | 0.9986362767 |
| 4 | auto | true | all_gather | true | 1086.94 | 0.995 | 29.440 | 1494.16 | 0.9986362767 |

## Interpretation

- The real hierarchical prototype is numerically safe in this short synthetic
  run: loss matches all-gather exactly.
- It is not a performance win on 4 A6000 GPUs:
  - `hierarchical_enabled` reaches only `45.7%` of all-gather throughput.
  - Step time grows from `29.285 ms` to `64.028 ms`.
- The likely cause is that the current hierarchical path adds three serialized
  communication phases per bucket:
  1. compressed local all-gather;
  2. uncompressed leader all-reduce;
  3. local broadcast.
- It does reduce peak memory from `1494.16 MB` to `922.91 MB`, because each rank
  dequantizes fewer compressed peer payloads. That is useful evidence for
  memory-oriented modes, but not enough to enable it for performance mode.
- `strategy="auto"` correctly protects throughput. Even with hierarchical
  transport available, it falls back to all-gather because the transport is not
  performance-recommended. The auto run keeps `99.5%` of all-gather throughput.

## Verification

Local focused tests:

```text
python -m pytest ccdl_comm_refactor/tests/test_hierarchical_transport.py \
  ccdl_comm_refactor/tests/test_hierarchical_api.py \
  ccdl_comm_refactor/tests/test_strategy_planner.py \
  ccdl_comm_refactor/tests/test_ddp_comm_hook.py \
  ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q
46 passed in 0.20s
```

A6000 container focused tests:

```text
PYTHONPATH=/workspace/ccdl_comm_refactor python -m pytest \
  tests/test_hierarchical_transport.py tests/test_hierarchical_api.py \
  tests/test_strategy_planner.py tests/test_ddp_comm_hook.py \
  tests/test_synthetic_ddp_script.py -q
46 passed in 1.45s
```

Raw JSON outputs are stored in `raw/`.

## Next performance direction

Do not tune this serialized hierarchical prototype as the default performance
path yet. The next performance-first step should be one of:

1. implement true compressed reduce-scatter semantics for DDP-compatible full
   bucket output, avoiding the extra uncompressed leader all-reduce;
2. fuse local compressed reduce and inter-group reduce in a native CUDA/C++
   scheduling path;
3. make hierarchical a memory-saving opt-in mode while keeping all-gather as the
   performance default.
