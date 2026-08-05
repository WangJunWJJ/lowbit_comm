# Correctness-first asynchronous kernel gate (A6000)

Date: 2026-08-05

## Scope

This report evaluates the same 44,971,744-parameter FP16 synthetic MLP on a
single five-GPU RTX A6000 host. Each run uses batch size 16 per rank, 22 total
steps, and 2 warm-up steps. The measured modes are native PyTorch DDP,
synchronous CCDL INT8 all-gather, and event-ordered asynchronous CCDL INT8
all-gather. Raw process output is preserved in [`raw/`](raw/).

The gate is deliberately correctness-first. Its schema-v2 inputs carry the full
mode-independent workload signature, which the gate requires to match exactly.
It then checks finite and decreasing loss, exact cross-rank parameter agreement,
convergence relative to native DDP, absence of fallback, and CUDA timeline
evidence before it considers a performance claim. The final performance
requirement is strict: asynchronous CCDL throughput must be greater than both
synchronous CCDL and native DDP under the same configuration.

## Results

| GPUs | Mode | Throughput (samples/s) | P50 step (ms) | P95 step (ms) | Peak allocated | Final loss | Relative to native |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | native DDP | 2144.54 | 14.878 | 15.163 | 402.3 MiB | 7.464570 | 1.000x |
| 2 | CCDL sync | 2240.10 | 13.949 | 15.029 | 922.7 MiB | 7.464570 | 1.045x |
| 2 | CCDL async | 2203.29 | 14.537 | 14.830 | 1011.4 MiB | 7.464570 | 1.027x |
| 4 | native DDP | 2990.11 | 21.300 | 21.854 | 402.3 MiB | 7.469248 | 1.000x |
| 4 | CCDL sync | 2267.00 | 28.100 | 28.420 | 1011.0 MiB | 7.469249 | 0.758x |
| 4 | CCDL async | 2150.28 | 29.772 | 30.153 | 1188.3 MiB | 7.469249 | 0.719x |

All six runs produced finite loss and exact cross-rank parameter agreement.
Sync and async CCDL also produced the same final loss at each world size. No
compressed run used a fallback.

## Gate decision

Both world sizes fail only at the final performance stage:

- 2 GPUs: async/sync = 0.9836x, so async is 1.64% slower than sync;
  it remains 2.74% faster than native DDP.
- 4 GPUs: async/sync = 0.9485x, so async is 5.15% slower than sync and
  28.09% slower than native DDP.

The CUDA timeline records non-zero overlap, but only 0.488% on 2 GPUs and
0.519% on 4 GPUs. This is evidence that the completion ordering works, not
evidence of useful compute/communication hiding. The current callback begins
communication near the end of backward, leaving too little independent compute
after each bucket launch. Additional callback, event, and workspace costs then
make the nominally asynchronous path slower.

The 4-GPU result also confirms that full-payload all-gather is the wrong scaling
transport: every rank receives and restores all compressed payloads, so traffic,
dequant-reduce work, and workspace grow with world size. Native NCCL all-reduce
therefore wins decisively at four ranks despite the smaller wire payload.

## Decision and next optimization boundary

Task 12 does not pass its performance acceptance gate. The implementation is
correct enough to continue optimization, but must not be advertised as an
asynchronous speedup. The next performance work should move communication to
earlier DDP buckets, reduce per-bucket Python/event overhead, and replace the
4+ rank full all-gather path with true compressed reduce-scatter plus a
ReducedShard consumer. Peak workspace usage also needs a separate budget gate.

## Provided policy code and 21 GB dataset

The provided `psi_policy` project was also run against the extracted
21,583,206,400-byte lake-view/Zarr archive. Its manifest contains 133 training
clips and 46,894 training frames. This is a real three-view 224x224 RGB policy
workload with a ViT observation encoder and an 8.9M-parameter DiT component.

The supplied archive is not standalone: it has no dependency manifest and
imports a missing `utils.transforms` module. The A6000 test used a pinned,
isolated Docker image and a test-only compatibility overlay; neither changes
the CCDL product package or the supplied policy source. The full data-backed
normalizer scan, model construction, checkpoint save, and validation path were
verified. Comparable performance runs used manual action bounds to avoid
repeating the approximately 175-second data-wide normalizer scan, disabled
pretrained weight download and `torch.compile`, and used the same settings for
all three communication modes.

Each comparable run executed 30 training steps at batch size 2 per rank. The
first 5 steps were excluded, leaving 25 measured steps:

| GPUs | Mode | Throughput (samples/s) | P50 step (ms) | P95 step (ms) | Validation loss | Relative to native |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | native DDP | 32.831 | 122.470 | 128.120 | 2.195991 | 1.000x |
| 2 | CCDL sync | 32.556 | 123.427 | 128.274 | 2.198915 | 0.992x |
| 2 | CCDL async | 32.057 | 125.940 | 131.250 | 2.199065 | 0.976x |
| 4 | native DDP | 60.798 | 132.011 | 139.443 | 1.646042 | 1.000x |
| 4 | CCDL sync | 60.481 | 133.258 | 136.709 | 1.646418 | 0.995x |
| 4 | CCDL async | 56.010 | 140.052 | 157.908 | 1.646129 | 0.921x |

The longer run reverses the apparent 10-step warm-up result: synchronous CCDL
is effectively tied but still slower than native DDP (0.84% on 2 GPUs and
0.52% on 4 GPUs), while asynchronous CCDL is 2.36% slower on 2 GPUs and 7.88%
slower on 4 GPUs. Validation loss differs from native by at most 0.14% on 2
GPUs and 0.023% on 4 GPUs, so no material short-run convergence regression was
observed. These 30 steps validate training behavior, not final task accuracy or
full convergence; a long multi-seed run is still required for that claim.

The structured result is preserved in
[`real_data_summary.json`](real_data_summary.json).

This real-data table is observational evidence rather than an input to the
formal correctness-first gate: the supplied trainer does not emit per-rank
parameter equality, CCDL fallback metadata, or CUDA overlap timeline fields.
The console logs confirmed the requested compressed `all_gather` hook was
attached, but that is weaker evidence than the synthetic gate schema. A future
business-model adapter must emit those fields before a real-data run can be
certified by the same gate.

## Reproduction

Run the gate after producing three JSON files with the same configuration:

```bash
python tests/benchmarks/run_e2e_overlap_gate.py \
  --native native.json --sync sync.json --async async.json \
  --output gate.json
```

Exit status `1` is expected for the raw results in this report because the
performance condition fails after all correctness conditions pass.
