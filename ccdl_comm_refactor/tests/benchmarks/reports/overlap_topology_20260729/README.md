# Native overlap topology validation, 2026-07-29

This report records the migration and A6000 validation of legacy CCDL
`qall_reduce(method="overlap-*")` methods into the refactored native
`ccdl_comm` topology transport.

## Coverage status

| Legacy method | Refactored native status | Notes |
| --- | --- | --- |
| `tree` | migrated | Native topology all-reduce; default for 2 GPUs. |
| `p2p` | migrated | Native topology all-reduce; still explicitly selectable. |
| `ring` | migrated | Native topology all-reduce; default for 4+ GPUs on tested A6000. |
| `overlap-gather` | migrated | Async compressed all-gather handle, then deferred remote dequant/reduce on `wait()`. |
| `overlap-p2p` | migrated | Preserves legacy behavior: ring reduce-scatter followed by overlapped compressed all-gather. |
| `overlap-tree` | migrated | Defers the final tree exchange completion to `wait()`. |
| `overlap-scale` | migrated | Preserves legacy scale all-reduce + int8 all-reduce async path. |
| `qreduce_scatter(method="ring"/"p2p")` | migrated | Available as `make_native_topology_reduce_scatter_shard`. |

The default DDP hook still calls topology transports with blocking completion
when `async_op=False`; the overlap methods are independently available and
validated, but full DDP Future/CUDA-stream overlap should be treated as a later
scheduling optimization.

## Environment and workload

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Model: FP16 synthetic MLP
- Parameters: `62,914,560`
- `input_dim=2048`
- `width=4096`
- `depth=4`
- `output_dim=1024`
- `batch_size_per_rank=16`
- `steps=20`
- `warmup_steps=5`
- Bucket cap: `512 MiB`
- Compression: INT8, group size 64

## Results

| GPUs | Method | Avg step ms | Samples/s | Train loss |
| ---: | --- | ---: | ---: | ---: |
| 2 | `overlap-gather` | 15.348 | 2084.97 | 0.9986624777 |
| 2 | `overlap-p2p` | 14.386 | 2224.34 | 0.9986624777 |
| 2 | `overlap-tree` | 13.312 | 2403.90 | 0.9986624777 |
| 2 | `overlap-scale` | 15.230 | 2101.06 | 0.9986624777 |
| 4 | `overlap-gather` | 33.808 | 1893.05 | 0.9972114623 |
| 4 | `overlap-p2p` | 19.119 | 3347.38 | 0.9972114623 |
| 4 | `overlap-tree` | 37.623 | 1701.10 | 0.9972114623 |
| 4 | `overlap-scale` | 19.937 | 3210.05 | 0.9972114623 |

## Interpretation

- The old topology surface is now functionally covered in the refactored native
  implementation.
- On this A6000 workload, `overlap-p2p` is the best 4-GPU overlap method and is
  close to the previously measured native ring auto path.
- `overlap-gather` and `overlap-tree` are not good default candidates for 4 GPUs
  in this benchmark because they add memory pressure or serialized exchange
  cost.
- `overlap-scale` is useful to preserve old CCDL behavior, but it should stay
  explicit until we add better numerical and convergence validation for the
  scale-reduce path.
