# End-to-end DDP training

`ddp_training.py` compares three modes with the same model, seed, sample order,
per-rank batch size, optimizer, and number of measured steps:

- `native_ddp`: unmodified PyTorch DDP;
- `ccdl_sync`: explicit synchronous INT8 all-gather hook;
- `ccdl_async`: event-ordered asynchronous INT8 all-gather hook.

The default MLP has 44,971,744 parameters. Synthetic samples are derived from
their global index, so repeated modes consume identical samples without storing
a generated dataset. For caller data, `--data-root` expects `.pt` files that
contain either `(input, target)` or `{"input": ..., "target": ...}`.

## CPU smoke

```bash
python examples/ddp_training.py --mode native_ddp --synthetic --device cpu \
  --steps 2 --warmup-steps 1 --batch-size-per-rank 2 --input-dim 32 \
  --hidden-dim 64 --depth 2 --num-classes 8 --output dist/example-smoke.json
```

## A6000 comparison

Run each mode separately so GPU clocks, global batch, and process placement are
comparable. A JSON config can provide defaults and explicit CLI flags override
them.

```bash
torchrun --standalone --nproc-per-node=2 examples/ddp_training.py \
  --config examples/configs/a6000_2gpu.json --mode native_ddp
torchrun --standalone --nproc-per-node=2 examples/ddp_training.py \
  --config examples/configs/a6000_2gpu.json --mode ccdl_sync \
  --output dist/a6000-2gpu-ccdl-sync.json
torchrun --standalone --nproc-per-node=2 examples/ddp_training.py \
  --config examples/configs/a6000_2gpu.json --mode ccdl_async \
  --output dist/a6000-2gpu-ccdl-async.json
```

Use `--nproc-per-node=4` with `a6000_4gpu.json` for the four-GPU run. Device-wide
synchronization is used only at measurement boundaries. For compressed DDP,
CUDA events record the backward interval and every bucket communication interval.
The JSON reports their union (`overlapped_ms`), actual intersection-derived
`overlap_efficiency`, and communication left exposed to the step critical path.
Returning a Future alone never produces the `timeline_overlapped` label.

Run the two-GPU dynamic oracle with:

```bash
torchrun --standalone --nproc-per-node=2 \
  tests/distributed/ddp_overlap_timeline.py
```

## Correctness-first performance gate

After producing comparable JSON files, require correctness and real asynchronous
timeline evidence before evaluating speedup:

```bash
python tests/benchmarks/run_e2e_overlap_gate.py \
  --native dist/a6000-2gpu-native.json \
  --sync dist/a6000-2gpu-ccdl-sync.json \
  --async dist/a6000-2gpu-ccdl-async.json \
  --output dist/a6000-2gpu-gate.json
```

The command first requires an identical mode-independent workload signature,
including model shape, optimizer inputs, seed, precision, compression settings,
and measured-step boundaries. It returns a non-zero status when comparability,
loss, rank consistency, compressed execution, asynchronous completion semantics,
or async-versus-sync/native throughput does not meet the gate. It still writes
the complete failure report so a slower result cannot be hidden. The current
A6000 2/4-GPU evidence and the additional
21 GB real-data policy run are documented in
[`tests/benchmarks/reports/correctness_async_kernel_20260805/README.md`](../tests/benchmarks/reports/correctness_async_kernel_20260805/README.md).
