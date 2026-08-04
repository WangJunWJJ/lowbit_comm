# Task 20: Device-resident dynamic metadata

## Scope

This report compares the existing Python-object metadata exchange
(object_v1) with the fixed 16 x int64 device packet (tensor_v1) used by
compiled dynamic all-gather.

- Host: user@192.168.8.156
- GPU: NVIDIA RTX A6000
- Container: ccdl-comm-a6000:cu126-torch25
- PyTorch: 2.5.0a0+872d972e41.nv24.08
- CUDA: 12.6
- Source commits: 88cbbf5 for explicit A/B measurement and c8f6124 for
  the gated auto selection
- Compression: FP16, INT8, group size 64
- Measurement: 20 warmup iterations and 1000 measured iterations per case

The payload target is approximate because an INT8/group64 payload grows in
66-byte units.

## Correctness

Both protocols passed 2-GPU and 4-GPU dynamic all-gather smoke tests.

| GPUs | Shapes | Large-shape max relative L2 | Boundary max relative L2 |
|---:|---|---:|---:|
| 2 | 524288, 524352 and 0/63/64/65 | 0.00595419 | 0.00394995 |
| 4 | 524288..524480 and 0/63/64/65 | 0.00595419 | 0.00391846 |

object_v1, explicit tensor_v1, and post-gate auto produced the same shape and
error results. The final auto smoke selected tensor_v1 on both 2 and 4 GPUs.

## Performance

The table reports the most conservative speedup across ranks. Values above
1.0 favor tensor_v1.

| GPUs | Payload target | CPU total p50 speedup | CPU total p95 speedup | GPU p50 speedup |
|---:|---:|---:|---:|---:|
| 2 | 1 KiB | 1.542x | 1.515x | 1.557x |
| 2 | 1 MiB | 1.495x | 1.489x | 1.500x |
| 2 | 16 MiB | 1.069x | 1.069x | 1.070x |
| 4 | 1 KiB | 1.652x | 1.733x | 1.668x |
| 4 | 1 MiB | 1.361x | 1.433x | 1.362x |
| 4 | 16 MiB | 1.039x | 1.039x | 1.039x |

The fixed packet therefore clears the G8 small-message requirement by a wide
margin and shows no 1 MiB or 16 MiB regression.

## Synchronization and launch evidence

Nsight Systems profiled like-for-like 2-GPU smoke runs:

| Protocol | cudaStreamSynchronize calls | NCCL all-gather kernel calls |
|---|---:|---:|
| object_v1 | 64 | 18 |
| tensor_v1 | 22 | 12 |

The tensor path removes pickle and all_gather_object; it performs one
contiguous metadata collective and one batched host decode. Dynamic output
allocation still requires host-visible shapes, but the new path introduces no
additional synchronization and reduced the observed synchronization count by
42.

The metadata workspace is allocated lazily per device and reused for steady
state. Observed peak-memory differences were generally 1 KiB; the largest
allocator variance was below 1 MiB and did not grow with iterations.

## Gate decision

G8 passes:

- protocol and boundary correctness passed;
- no additional GPU-to-CPU synchronization was observed;
- 2-GPU and 4-GPU small-message latency improved by more than 10%;
- 1 MiB and 16 MiB cases did not regress;
- packet workspaces are caller-owned and reused in steady state.

Explicit auto is therefore capability-gated to tensor_v1 when
all_gather_into_tensor is available. Unsupported backends retain the
object_v1 fallback and a diagnostic fallback reason. The API default remains
object_v1, so existing callers do not change behavior without opting into
auto or tensor_v1.

## Build warnings resolved after validation

The canonical CUDA build flag is CCDL_COMM_BUILD_CUDA=1 and all active plan
commands now use it. CCDL_BUILD_CUDA remains a compatibility alias, while an
explicit canonical value takes precedence. The setup entry point also
bootstraps its source root, so editable metadata and extension builds no
longer require a caller-provided PYTHONPATH.

## Evidence

Raw JSON and profiler CSV files are stored in raw/. Full .nsys-rep traces
remain on the isolated A6000 validation worktree because they are large binary
artifacts.
