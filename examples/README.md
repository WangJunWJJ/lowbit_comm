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
synchronization is currently used only at measurement boundaries; Task 11 adds
CUDA-event timeline evidence for real compute/communication overlap.
