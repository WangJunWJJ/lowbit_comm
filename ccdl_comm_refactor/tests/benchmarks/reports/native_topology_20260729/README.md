# Native topology transport validation, 2026-07-29

This report validates the second performance-first migration step: moving the
legacy topology-aware tree/p2p algorithms into native `ccdl_comm` transport code.
The default `strategy="topology"` path no longer imports the legacy `ccdl.comm`
package; `make_legacy_topology_all_reduce` remains as a compatibility alias to
the native implementation.

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
- Compression: INT8, group size 64, top-k 0

## Results

| Version / path | GPUs | Avg step ms | Samples/s | Train loss | Throughput vs legacy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy CCDL tree | 2 | 13.247 | 2415.72 | 0.9986604303 | 100.0% |
| Refactor topology bridge | 2 | 13.622 | 2349.14 | 0.9986624777 | 97.2% |
| Refactor native topology | 2 | 13.265 | 2412.40 | 0.9986624777 | 99.9% |
| Legacy CCDL p2p | 4 | 20.808 | 3075.72 | 0.9972144723 | 100.0% |
| Refactor topology bridge | 4 | 20.926 | 3058.39 | 0.9972114623 | 99.4% |
| Refactor native topology | 4 | 21.017 | 3045.11 | 0.9972114623 | 99.0% |

## Interpretation

- The native refactor recovers the legacy topology performance envelope:
  - 2 GPUs: `2412.40 / 2415.72 = 99.9%` of legacy throughput.
  - 4 GPUs: `3045.11 / 3075.72 = 99.0%` of legacy throughput.
- The training loss remains aligned with the legacy and bridge runs at this
  short synthetic scale, which is the expected result because all paths use the
  same INT8 compression configuration and synchronous replicated gradient mean.
- The native implementation removes the legacy package import from the default
  topology path while retaining the refactored safety contract and testable
  injection points.

This closes the first migration objective: old CCDL topology behavior is now
available through the new `ccdl_comm` API without depending on old Python
modules, and it does not regress materially against the pre-refactor baseline on
the measured 2-GPU and 4-GPU A6000 cases.
