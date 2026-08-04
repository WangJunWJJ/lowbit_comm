# A6000 CUDA Completion Integration Benchmark

## Environment

- Host: five-GPU NVIDIA RTX A6000 server
- Validation ranks: 2 and 4
- Container: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: 12.6
- Tensor: 4,194,304 FP16 elements per rank
- Compression: INT8, group size 64
- Warm-up: 10 iterations
- Measurement: 30 iterations
- Overlap workload: 64 independent in-place CUDA `sin` kernels

The synchronous and asynchronous cases use identical tensors and compression
settings. `async_wait_ms` includes launch, `wait()`, deferred dequantization,
reduction, and final CUDA completion. `async_overlap_ms` inserts the independent
CUDA workload between launch and `wait()`.

## Results

| Path | GPUs | Sync ms | Async launch+wait ms | Async speedup | Compute ms | Async+compute ms | Overlap efficiency | Relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-gather-reduce | 2 | 0.9603 | 0.9582 | 1.002x | 0.3467 | 1.0949 | 61.2% | 0.005936 |
| all-gather-reduce | 4 | 2.3209 | 2.3153 | 1.002x | 0.3424 | 2.3490 | 91.8% | 0.005952 |
| topology overlap-gather | 2 | 0.9445 | 0.9440 | 1.001x | 0.3397 | 1.1216 | 47.9% | 0.005936 |
| topology overlap-p2p | 4 | 1.5482 | 1.3644 | 1.135x | 0.4664 | 1.5539 | 98.8% | 0.009397 |

The raw asynchronous launch latency was 201-248 microseconds for the
all-gather and overlap-gather paths. The four-rank overlap-p2p path spent about
948 microseconds before returning its Work because its reduce-scatter phase is
still serialized before the final asynchronous all-gather.

## Interpretation

- The unified Work layer does not materially penalize the validated
  all-gather path: two- and four-rank launch-plus-wait latency differs from the
  synchronous path by less than 0.3%.
- Four-rank all-gather hides about 92% of the selected independent compute
  workload. This is the clearest evidence that the returned Work now provides
  useful overlap rather than only an asynchronous-looking API.
- Four-rank overlap-p2p improves launch-plus-wait latency by about 13.5% and
  hides about 99% of the selected compute workload.
- Two-rank topology overlap-gather exposes less useful overlap than the
  four-rank cases because communication is already short relative to launch,
  callback, and kernel overhead.
- The four-rank overlap-p2p relative L2 error is higher because its
  reduce-scatter and gather stages apply compression more than once. It remains
  a topology-specific accuracy trade-off and should not silently replace the
  default all-gather strategy.

## Point-to-point validation

The updated blocking and asynchronous P2P APIs were also run with 4,194,304
FP16 elements on two A6000 GPUs:

- Blocking relative L2: `0.00593529`
- Asynchronous relative L2: `0.00593529`

The identical error confirms that routing blocking P2P through the unified
result-bearing Work did not change reconstruction accuracy or introduce a
deadlock.

## Raw data

- `raw/2gpu_all_gather_reduce.json`
- `raw/4gpu_all_gather_reduce.json`
- `raw/2gpu_topology_overlap_gather.json`
- `raw/4gpu_topology_overlap_p2p.json`
- `raw/2gpu_p2p_smoke.json`
