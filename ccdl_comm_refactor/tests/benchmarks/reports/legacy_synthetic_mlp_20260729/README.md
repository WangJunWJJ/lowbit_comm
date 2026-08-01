# Legacy CCDL FP16 synthetic MLP benchmark, 2026-07-29

This report records the pre-refactor `ccdl.comm.qall_reduce` synthetic MLP
training benchmark requested for 2-GPU and 4-GPU A6000 runs.

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
- Compression: INT8, group size 64, top-k 0

The benchmark uses replicated local training and manually synchronizes flattened
gradients with legacy CCDL:

- 2 GPUs: `qall_reduce(..., method="tree")`
- 4 GPUs: `qall_reduce(..., method="p2p")`

## Results

| GPUs | Legacy method | Global batch | Avg step ms | Samples/s | Train loss |
| ---: | --- | -----------: | ----------: | --------: | ---------: |
| 2 | tree | 32 | 13.247 | 2415.72 | 0.9986604303 |
| 4 | p2p | 64 | 20.808 | 3075.72 | 0.9972144723 |

The 4-GPU throughput is `1.27x` the 2-GPU throughput in this run.
