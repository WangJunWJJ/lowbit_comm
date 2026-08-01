# GPU-first A6000 baseline

## Scope

This report freezes the Task 0 latency, memory, throughput, and numerical-error
baseline before the new Core and compiled-executor architecture is introduced.
The raw JSON files are machine checked by `tests/test_performance_gate.py`.

## Environment

- Source commit: `7850148`
- Host: `user-SYS-6049GP-TRT-LongJing-Server`
- GPU: 5 × NVIDIA RTX A6000 48 GiB; tests use 2 or 4 GPUs
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA runtime: `12.6`
- CUDA extension: locally built for `sm_86`
- Quantization: INT8, group size 64, non-compact payload
- Timing: 20 warm-up iterations and 100 measured iterations
- Effective GB/s is logical tensor bytes divided by latency, not physical wire
  throughput.

The extension was built with:

```bash
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
  python setup.py build_ext --inplace
```

The container must receive the project root in `PYTHONPATH`, plus stable source
and host identity because a bind-mounted Git worktree can be rejected by Git's
safe-directory check and the container hostname is ephemeral:

```bash
docker run --rm --gpus all --shm-size=8g \
  -e PYTHONPATH=/workspace/ccdl_comm_refactor \
  -e CCDL_BENCHMARK_COMMIT=7850148 \
  -e CCDL_BENCHMARK_HOSTNAME=user-SYS-6049GP-TRT-LongJing-Server \
  -v /home/user/wangjun/lowbit_comm:/workspace \
  -w /workspace/ccdl_comm_refactor \
  ccdl-comm-a6000:cu126-torch25 \
  bash -lc 'for nproc in 2 4; do
    for dtype in fp16 bf16; do
      for numel in 524288 8388608 33554432; do
        torchrun --standalone --nproc-per-node=${nproc} \
          tests/distributed/collective_perf_compare.py \
          --dtype=${dtype} --numel=${numel} --bit=8 --group-size=64 \
          --warmup=20 --repeat=100 \
          --output-json=tests/benchmarks/reports/gpu_first_baseline/raw/${nproc}gpu_${dtype}_${numel}.json
      done
    done
  done'
```

## All-reduce results

The ratio is CCDL compressed all-gather-reduce latency divided by native
PyTorch all-reduce latency; lower is better.

| GPUs | dtype | elements | PyTorch ms | CCDL ms | ratio | relative L2 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 2 | FP16 | 524,288 | 0.1820 | 0.3838 | 2.109 | 0.005927 |
| 2 | FP16 | 8,388,608 | 2.4879 | 1.8089 | 0.727 | 0.005937 |
| 2 | FP16 | 33,554,432 | 9.8736 | 6.9935 | 0.708 | 0.005940 |
| 2 | BF16 | 524,288 | 0.1818 | 0.3841 | 2.113 | 0.006568 |
| 2 | BF16 | 8,388,608 | 2.4784 | 1.8108 | 0.731 | 0.006585 |
| 2 | BF16 | 33,554,432 | 9.8990 | 7.0065 | 0.708 | 0.006586 |
| 4 | FP16 | 524,288 | 0.2561 | 0.4479 | 1.749 | 0.005949 |
| 4 | FP16 | 8,388,608 | 3.6049 | 4.4369 | 1.231 | 0.005949 |
| 4 | FP16 | 33,554,432 | 14.2818 | 17.2397 | 1.207 | 0.005953 |
| 4 | BF16 | 524,288 | 0.2528 | 0.4585 | 1.814 | 0.007174 |
| 4 | BF16 | 8,388,608 | 3.5971 | 4.4271 | 1.231 | 0.007170 |
| 4 | BF16 | 33,554,432 | 14.2699 | 17.1817 | 1.204 | 0.007173 |

All 48 standardized result records contain zero non-finite values. The 2-GPU
path wins for medium and large tensors, while the current 4-GPU all-gather
implementation regresses. This is a baseline fact, not an accepted performance
target: later tasks must route 4-GPU traffic toward compressed reduce-scatter or
another topology that avoids gathering every rank's complete payload.

## Problems found and closed

1. Direct `torchrun` initially could not import `ccdl_comm`. Root cause: unlike
   pytest, direct script execution did not add the project root to `sys.path`.
   The reproducible container command now sets `PYTHONPATH` explicitly.
2. The fresh checkout did not contain `ccdl_cuda_ops`. It was built with the
   documented opt-in CUDA build and passed all three extension smoke tests.
3. The first valid JSON used an ephemeral container hostname and an `unknown`
   commit. The result schema now rejects an unknown revision, and benchmark
   identity supports explicit container-safe source and host overrides.

No Traceback, benchmark ERROR, NCCL WARN, or non-finite value remained in the
final 12-run matrix.
