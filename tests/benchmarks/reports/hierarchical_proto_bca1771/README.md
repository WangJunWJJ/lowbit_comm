# A6000 hierarchical transport prototype validation, 2026-07-28

## Scope

This report validates the capability-gated hierarchical compressed transport
prototype. The goal of this phase is semantic safety:

- an injected fake/local hierarchical transport can be called by unit tests;
- production DDP defaults do not enable an unavailable hierarchical transport;
- `strategy="hierarchical"` falls back to the validated all-gather path when no
  real transport is provided;
- benchmark JSON explains selected strategy and fallback status.

This report does not claim a new hierarchical performance fast path.

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

## Result

| GPUs | Requested strategy | Selected strategy | Requires fallback | Fallback reason | Samples/s | Avg step ms | Loss |
|---:|---|---|---|---|---:|---:|---:|
| 4 | all_gather | all_gather | false | explicit strategy all_gather | 1088.72 | 29.392 | 0.9986362767 |
| 4 | auto | all_gather | true | reduce_scatter unavailable for single-node auto strategy; falling back to all_gather | 1091.35 | 29.322 | 0.9986362767 |
| 4 | hierarchical | all_gather | true | hierarchical transport unavailable for explicit strategy; falling back to all_gather | 1093.80 | 29.256 | 0.9986362767 |

## Interpretation

- `strategy="hierarchical"` is now accepted by the DDP benchmark path, but it is
  capability-gated.
- Without an injected real hierarchical transport, the DDP hook reports
  `selected_strategy="all_gather"` and `strategy_requires_fallback=true`.
- Loss is identical across explicit all-gather, auto fallback, and hierarchical
  fallback for this short synthetic run.
- Step-time differences among the three fallback runs are within normal
  short-run noise. This confirms the prototype did not introduce a correctness
  regression or an obvious fallback overhead on 4 GPUs.

## Verification

Local focused tests:

```text
python -m pytest ccdl_comm_refactor/tests/test_hierarchical_api.py \
  ccdl_comm_refactor/tests/test_strategy_planner.py \
  ccdl_comm_refactor/tests/test_ddp_comm_hook.py \
  ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q
41 passed in 0.13s
```

A6000 container focused tests:

```text
PYTHONPATH=/workspace/ccdl_comm_refactor python -m pytest \
  tests/test_hierarchical_api.py tests/test_strategy_planner.py \
  tests/test_ddp_comm_hook.py tests/test_synthetic_ddp_script.py -q
41 passed in 1.51s
```

Raw JSON outputs are stored in `raw/`.

## Next step

The next implementation step should replace the fake/injected hierarchical
transport with a real capability-gated prototype that constructs local and
cross-group process groups. It should still default to fallback until same-shape
4-GPU and, later, 8-GPU/multi-node results prove it is faster and numerically
safe.
