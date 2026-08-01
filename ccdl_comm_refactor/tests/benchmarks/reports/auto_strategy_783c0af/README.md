# A6000 auto strategy planner validation, 2026-07-28

## Scope

This report validates the first topology-aware strategy-planning step. The new
`strategy="auto"` path is expected to select an explainable strategy and safely
fall back to the current validated all-gather path because compressed
reduce-scatter and hierarchical transports are not enabled yet.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
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

| GPUs | Requested strategy | Selected strategy | Fallback reason | Samples/s | Avg step ms | Loss |
|---:|---|---|---|---:|---:|---:|
| 2 | all_gather | all_gather | explicit strategy all_gather | 1039.16 | 15.397 | 0.9980579770 |
| 2 | auto | all_gather | world_size<=2 uses validated all_gather path | 1019.09 | 15.700 | 0.9980579770 |
| 4 | all_gather | all_gather | explicit strategy all_gather | 1096.33 | 29.188 | 0.9986362767 |
| 4 | auto | all_gather | reduce_scatter unavailable for single-node auto strategy; falling back to all_gather | 1088.03 | 29.411 | 0.9986362767 |

## Interpretation

- `strategy="auto"` does not enable an unimplemented compressed reduce-scatter
  transport. It falls back to all-gather with explicit metadata.
- The short-run synthetic loss is identical between explicit all-gather and
  auto fallback for both 2-GPU and 4-GPU runs.
- The observed auto fallback overhead is small in this run:
  - 2 GPU: samples/s ratio `0.981`, step time `+1.97%`
  - 4 GPU: samples/s ratio `0.992`, step time `+0.76%`
- This validates the planner/reporting layer, not a new performance fast path.
  The next performance step is to implement and capability-gate a real
  compressed reduce-scatter or hierarchical transport.

## Local and remote verification

Local focused tests:

```text
python -m pytest ccdl_comm_refactor/tests/test_strategy_planner.py \
  ccdl_comm_refactor/tests/test_reduce_scatter_api.py \
  ccdl_comm_refactor/tests/test_ddp_comm_hook.py \
  ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q
37 passed in 0.17s
```

A6000 container focused tests:

```text
PYTHONPATH=/workspace/ccdl_comm_refactor python -m pytest \
  tests/test_strategy_planner.py tests/test_reduce_scatter_api.py \
  tests/test_ddp_comm_hook.py tests/test_synthetic_ddp_script.py -q
37 passed in 1.49s
```

Raw JSON outputs are stored in `raw/`.
