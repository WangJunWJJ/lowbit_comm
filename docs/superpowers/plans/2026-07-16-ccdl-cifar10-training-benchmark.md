# CCDL CIFAR-10 Training Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible 2-GPU CIFAR-10/ResNet-18 experiment that compares NCCL FP32 gradient synchronization with CCDL INT8/INT4 Top-K communication, including communication speed, training accuracy, and convergence-step analysis.

**Architecture:** A self-contained `benchmarks/cifar10` package owns configuration, model/data construction, flat-gradient synchronization, training, microbenchmarks, aggregation, and reporting. Training uses an identical contiguous gradient buffer for the NCCL and CCDL paths so synchronization timing and convergence are comparable; raw JSONL logs are the source of truth for all summaries and plots.

**Tech Stack:** Python 3.10+, PyTorch/CUDA/NCCL, torchvision, CCDL CUDA extension, pytest, matplotlib, JSONL/CSV, SSH to `wangjun@192.168.1.100`.

## Global Constraints

- Run on `wangjun@192.168.1.100` using exactly 2×NVIDIA GeForce RTX 4090 D.
- Use CIFAR-10 and a CIFAR-adapted ResNet-18 with a 3×3 stem and no initial max-pool.
- Run 200 epochs and seeds `1337`, `2027`, and `4099` for every main configuration.
- Main configurations are NCCL-FP32, CCDL INT8 K0/K2, and CCDL INT4 K0/K2; CCDL uses group size 64 and deterministic rounding.
- Convergence means first reaching 99% of the matching seed's NCCL final Top-1 and staying at or above it for 5 consecutive epochs.
- Preserve all raw logs and environment/source hashes; do not claim results from a failed or partial run.
- The workspace has no Git metadata. Replace commit steps with immutable SHA-256 source manifests and timestamped run directories; never claim a Git commit was created.

---

## File Structure

- `benchmarks/cifar10/config.py`: immutable experiment configuration and matrix expansion.
- `benchmarks/cifar10/model.py`: CIFAR-10 ResNet-18 construction.
- `benchmarks/cifar10/data.py`: deterministic distributed CIFAR-10 loaders.
- `benchmarks/cifar10/sync.py`: flat gradient packing and NCCL/CCDL synchronization.
- `benchmarks/cifar10/logging_utils.py`: rank-safe JSONL logging and CUDA phase timing.
- `benchmarks/cifar10/train.py`: distributed training/evaluation entry point.
- `benchmarks/cifar10/comm_bench.py`: isolated NCCL/CCDL communication benchmark.
- `benchmarks/cifar10/smoke.py`: CUDA round-trip and two-rank numerical checks.
- `benchmarks/cifar10/aggregate.py`: convergence and three-seed statistics.
- `benchmarks/cifar10/plot_report.py`: figures and Chinese Markdown report.
- `benchmarks/cifar10/run_matrix.sh`: sequential, resumable 15-run launcher.
- `tests/benchmarks/cifar10/`: CPU unit tests for configuration, packing, convergence, and aggregation.

### Task 1: Configuration and deterministic experiment matrix

**Files:**
- Create: `benchmarks/cifar10/__init__.py`
- Create: `benchmarks/cifar10/config.py`
- Create: `tests/benchmarks/cifar10/test_config.py`

**Interfaces:**
- Produces: `RunConfig`, `MAIN_VARIANTS`, `SEEDS`, and `expand_main_matrix(output_root: Path) -> list[RunConfig]`.

- [ ] **Step 1: Write failing matrix tests**

```python
from pathlib import Path
from benchmarks.cifar10.config import expand_main_matrix

def test_main_matrix_has_fifteen_unique_runs(tmp_path: Path):
    runs = expand_main_matrix(tmp_path)
    assert len(runs) == 15
    assert len({r.run_id for r in runs}) == 15
    assert {r.seed for r in runs} == {1337, 2027, 4099}
    assert {r.variant for r in runs} == {
        "nccl_fp32", "ccdl_int8_k0", "ccdl_int8_k2",
        "ccdl_int4_k0", "ccdl_int4_k2",
    }

def test_quantized_variants_are_fixed_to_group64_deterministic(tmp_path: Path):
    for run in expand_main_matrix(tmp_path):
        if run.variant.startswith("ccdl"):
            assert run.group_size == 64
            assert run.stochastic is False
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/benchmarks/cifar10/test_config.py -q`

Expected: FAIL because `benchmarks.cifar10.config` does not exist.

- [ ] **Step 3: Implement the exact matrix**

```python
from dataclasses import dataclass
from pathlib import Path

SEEDS = (1337, 2027, 4099)
MAIN_VARIANTS = {
    "nccl_fp32": (None, None),
    "ccdl_int8_k0": (8, 0),
    "ccdl_int8_k2": (8, 2),
    "ccdl_int4_k0": (4, 0),
    "ccdl_int4_k2": (4, 2),
}

@dataclass(frozen=True)
class RunConfig:
    variant: str
    seed: int
    output_dir: Path
    epochs: int = 200
    batch_size_per_rank: int = 128
    workers_per_rank: int = 4
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    bit: int | None = None
    topk: int | None = None
    group_size: int = 64
    stochastic: bool = False

    @property
    def run_id(self) -> str:
        return f"{self.variant}-seed{self.seed}"

def expand_main_matrix(output_root: Path) -> list[RunConfig]:
    return [
        RunConfig(name, seed, output_root / f"{name}-seed{seed}", bit=bit, topk=topk)
        for seed in SEEDS
        for name, (bit, topk) in MAIN_VARIANTS.items()
    ]
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/benchmarks/cifar10/test_config.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Record source state**

Run: `Get-ChildItem benchmarks/cifar10,tests/benchmarks/cifar10 -Recurse -File | Get-FileHash -Algorithm SHA256`

Expected: SHA-256 entry for every created file; save later in the remote run manifest.

### Task 2: Model, data, and flat-gradient synchronization

**Files:**
- Create: `benchmarks/cifar10/model.py`
- Create: `benchmarks/cifar10/data.py`
- Create: `benchmarks/cifar10/sync.py`
- Create: `tests/benchmarks/cifar10/test_model_sync.py`

**Interfaces:**
- Produces: `build_model(num_classes=10) -> nn.Module`.
- Produces: `build_loaders(data_root, config, rank, world_size) -> tuple[DataLoader, DataLoader, DistributedSampler]`.
- Produces: `FlatGradientSynchronizer(model, mode, bit, topk, group_size)` with `pack()`, `synchronize()`, and `unpack()`.

- [ ] **Step 1: Write CPU tests for the CIFAR stem and gradient round-trip**

```python
import torch
from benchmarks.cifar10.model import build_model
from benchmarks.cifar10.sync import FlatGradientSynchronizer

def test_resnet18_uses_cifar_stem():
    model = build_model()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

def test_pack_unpack_preserves_all_gradients_without_padding():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    model(torch.randn(5, 4)).sum().backward()
    expected = [p.grad.clone() for p in model.parameters()]
    sync = FlatGradientSynchronizer(model, mode="nccl_fp32")
    flat = sync.pack()
    for p in model.parameters():
        p.grad.zero_()
    sync.unpack(flat)
    assert all(torch.equal(p.grad, e) for p, e in zip(model.parameters(), expected))
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/benchmarks/cifar10/test_model_sync.py -q`

Expected: FAIL because model and synchronization modules do not exist.

- [ ] **Step 3: Implement model and deterministic loaders**

Use `torchvision.models.resnet18(weights=None, num_classes=10)`, replace `conv1` with `nn.Conv2d(3,64,3,1,1,bias=False)`, and replace `maxpool` with `nn.Identity()`. Train transforms must be `RandomCrop(32,padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, and CIFAR-10 normalization; validation uses only `ToTensor()` and normalization. Construct the training sampler with `DistributedSampler(..., shuffle=True, seed=config.seed)` and call `sampler.set_epoch(epoch)` in training.

- [ ] **Step 4: Implement the synchronizer**

`pack()` must concatenate only non-None gradients into one contiguous FP32 CUDA tensor and retain `(parameter, offset, numel, shape)` views. For `nccl_fp32`, `synchronize()` calls `dist.all_reduce(flat, SUM)` then divides by world size. For CCDL modes, construct `Quantizer(group_size, -1, bit, topk, False, "fp32")`, call `qall_reduce(flat, op="mean", method="tree", keep_self=False, async_op=False)`, and return the synchronized buffer. `unpack()` copies each slice back to its original gradient without changing shape.

- [ ] **Step 5: Run unit tests and syntax checks**

Run: `python -m pytest tests/benchmarks/cifar10/test_model_sync.py -q && python -m compileall -q benchmarks/cifar10`

Expected: `2 passed` and exit code 0.

### Task 3: Training, evaluation, timing, and structured logging

**Files:**
- Create: `benchmarks/cifar10/logging_utils.py`
- Create: `benchmarks/cifar10/train.py`
- Create: `tests/benchmarks/cifar10/test_logging.py`

**Interfaces:**
- Consumes: `RunConfig`, `build_model`, `build_loaders`, `FlatGradientSynchronizer`.
- Produces: one `metrics.jsonl`, `epochs.csv`, `config.json`, `environment.json`, and checkpoint per run.

- [ ] **Step 1: Test rank-safe JSONL emission**

```python
import json
from benchmarks.cifar10.logging_utils import JsonlLogger

def test_jsonl_logger_writes_one_valid_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path, rank=0).emit("epoch", epoch=3, val_top1=81.25)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row == {"kind": "epoch", "epoch": 3, "val_top1": 81.25}

def test_nonzero_rank_does_not_write(tmp_path):
    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path, rank=1).emit("epoch", epoch=0)
    assert not path.exists()
```

- [ ] **Step 2: Run the failing logger test**

Run: `python -m pytest tests/benchmarks/cifar10/test_logging.py -q`

Expected: FAIL because `JsonlLogger` does not exist.

- [ ] **Step 3: Implement logger and CUDA phase timer**

`JsonlLogger.emit(kind, **fields)` must append one UTF-8 JSON object and flush immediately on rank 0 only. `CudaTimer.measure(name, callable)` must record start/end CUDA events on the current stream, synchronize the end event, and return `(callable_result, elapsed_ms)`.

- [ ] **Step 4: Implement distributed training**

Use SGD with momentum 0.9, weight decay `5e-4`, initial LR 0.2, and cosine decay over 200 epochs. Broadcast initial model parameters from rank 0 before training. At every step measure forward, backward, gradient synchronization, optimizer, and total time separately; exclude the first 20 steps from throughput summaries. Evaluate the full validation set every epoch using distributed sums of loss, correct predictions, and sample count. Log gradient norm, finite-state checks, CUDA peak memory, images/s, cumulative optimizer steps, and wall time. Save `last.pt` after each epoch and support `--resume` without repeating completed epochs.

- [ ] **Step 5: Validate CLI without GPU work**

Run: `python -m benchmarks.cifar10.train --help`

Expected: help text containing `--variant`, `--seed`, `--data-root`, `--output-dir`, and `--resume`.

- [ ] **Step 6: Run all CPU tests**

Run: `python -m pytest tests/benchmarks/cifar10 -q`

Expected: all tests pass.

### Task 4: CUDA and two-rank preflight verification

**Files:**
- Create: `benchmarks/cifar10/smoke.py`
- Create: `benchmarks/cifar10/comm_bench.py`

**Interfaces:**
- Produces: `preflight.jsonl` with Q/DQ and all-reduce errors.
- Produces: `comm.jsonl` with raw repeat latency samples and stage timings.

- [ ] **Step 1: Implement single-GPU round-trip checks**

For `(bit, topk)` in `(8,0),(8,2),(4,0),(4,2)`, generate a deterministic FP32 CUDA tensor whose length is divisible by 64, quantize/dequantize it, assert finite output and exact expected compressed length from `Quantizer.get_lenq`, and record relative L2/RMSE/max-absolute error.

- [ ] **Step 2: Implement two-rank all-reduce checks**

Initialize CCDL/NCCL, create rank-dependent deterministic tensors, compute a native FP32 SUM reference, run each CCDL configuration on a clone, and record relative L2/RMSE/max error. Fail if any output is non-finite or if INT8 relative L2 exceeds `0.02` or INT4 relative L2 exceeds `0.25`.

- [ ] **Step 3: Implement communication microbenchmark**

Test FP32-equivalent sizes 1, 4, 16, 64, and 256 MiB. Use 20 warmups and 100 measured repeats per configuration. Store each repeat rather than only averages. NCCL timing must include only `dist.all_reduce`; CCDL total timing includes quantize, compressed communication, and dequantize, with separate CUDA-event stages where the API permits. Compute compression from actual `q.numel()`, not theoretical bit width.

- [ ] **Step 4: Copy the benchmark package to a timestamped remote directory**

Run locally:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
ssh wangjun@192.168.1.100 "mkdir -p ~/work/ccdl_cifar10_runs/$stamp/src"
scp -r benchmarks tests pyproject.toml setup.py ccdl csrc wangjun@192.168.1.100:work/ccdl_cifar10_runs/$stamp/src/
```

Expected: all source directories appear under one immutable timestamped run root.

- [ ] **Step 5: Create remote SHA-256 manifest**

Run remotely inside the timestamped source root:

```bash
find . -type f -print0 | sort -z | xargs -0 sha256sum > source-sha256.txt
nvidia-smi -q > gpu-environment.txt
python -m torch.utils.collect_env > torch-environment.txt
```

Expected: three non-empty environment/manifest files.

- [ ] **Step 6: Build CCDL in the existing CUDA-capable runtime**

Use the same CUDA-capable container/runtime that produced `~/work/ccdl_bench/*.jsonl`, mount the timestamped source and `~/work/datasets`, then run:

```bash
python -m pip install --no-build-isolation -e .
python -c 'import torch, ccdl, ccdl_cuda_ops; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

Expected: import succeeds and reports 2 CUDA devices. If the historical runtime cannot be recovered, stop and record the exact missing runtime instead of silently changing CUDA/PyTorch versions.

- [ ] **Step 7: Run preflight and microbenchmark**

Run inside the verified runtime:

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarks.cifar10.smoke --output results/preflight.jsonl
torchrun --standalone --nproc-per-node=2 -m benchmarks.cifar10.comm_bench --output results/comm.jsonl
```

Expected: both exit 0, preflight contains 4 Q/DQ and 4 all-reduce records, and comm output contains all 25 variant/size combinations.

### Task 5: Short training smoke tests and resumable full matrix

**Files:**
- Create: `benchmarks/cifar10/run_matrix.sh`

**Interfaces:**
- Consumes: training CLI and expanded matrix.
- Produces: 15 isolated run directories plus `matrix-status.jsonl`.

- [ ] **Step 1: Implement a strict launcher**

The script must use `set -euo pipefail`, accept `DATA_ROOT` and `OUTPUT_ROOT`, run NCCL baselines before matching quantized configurations, skip only runs containing a valid `complete.json`, and append start/success/failure records to `matrix-status.jsonl`. Each launch uses:

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarks.cifar10.train \
  --variant "$variant" --seed "$seed" --epochs "$epochs" \
  --data-root "$DATA_ROOT" --output-dir "$OUTPUT_ROOT/$variant-seed$seed" --resume
```

- [ ] **Step 2: Run five 2-epoch smoke tests for seed 1337**

Run every main variant for 2 epochs. Expected: every process exits 0, losses and gradients remain finite, epoch 1 training loss is below or statistically consistent with epoch 0, and each directory contains config, environment, JSONL, CSV, and checkpoint files.

- [ ] **Step 3: Inspect smoke-test comparability**

Verify identical initial parameter SHA-256, identical first-epoch sampler indices, identical global batch size, and matching optimizer/LR fields across all five runs. Expected: only communication-related fields differ.

- [ ] **Step 4: Launch the 15-run, 200-epoch matrix**

Run:

```bash
RUN_ROOT=$(cat ~/work/ccdl_cifar10_runs/CURRENT_RUN) \
DATA_ROOT=~/work/datasets/cifar10 \
OUTPUT_ROOT="$RUN_ROOT/results/train" \
bash benchmarks/cifar10/run_matrix.sh
```

Expected: 15 `complete.json` files. Monitor GPU utilization, disk space, non-finite metrics, and process exits; preserve logs on any failure and resume only through the documented checkpoint path.

### Task 6: Convergence calculation, aggregation, figures, and report

**Files:**
- Create: `benchmarks/cifar10/aggregate.py`
- Create: `benchmarks/cifar10/plot_report.py`
- Create: `tests/benchmarks/cifar10/test_convergence.py`

**Interfaces:**
- Produces: `find_convergence(epochs, threshold, patience=5) -> dict | None`.
- Produces: `summary.json`, `summary.csv`, PNG figures, and `CCDL_CIFAR10_REPORT_ZH.md`.

- [ ] **Step 1: Write convergence tests**

```python
from benchmarks.cifar10.aggregate import find_convergence

def test_convergence_requires_five_consecutive_epochs():
    rows = [{"epoch": i, "val_top1": v, "optimizer_steps": 100*(i+1), "wall_s": 10*(i+1)}
            for i, v in enumerate([79, 80, 79, 80, 80, 80, 80, 80])]
    result = find_convergence(rows, threshold=80.0, patience=5)
    assert result == {"epoch": 3, "optimizer_steps": 400, "wall_s": 40}

def test_convergence_returns_none_when_not_sustained():
    rows = [{"epoch": i, "val_top1": v, "optimizer_steps": i, "wall_s": i}
            for i, v in enumerate([80, 80, 80, 80, 79])]
    assert find_convergence(rows, threshold=80.0, patience=5) is None
```

- [ ] **Step 2: Verify failure, implement, and rerun**

Run before implementation: `python -m pytest tests/benchmarks/cifar10/test_convergence.py -q`

Expected before: import failure. Implement a sliding window that returns the first row of the first five-row window whose `val_top1 >= threshold` throughout. Run again; expected `2 passed`.

- [ ] **Step 3: Aggregate all runs**

For each seed, read the NCCL final Top-1 and set threshold to `0.99 * final_top1`. Apply that seed-specific threshold to all five variants. Compute per-seed best/final/last-5 Top-1, convergence epoch/steps/wall time, steady-state images/s, communication share, peak memory, and gradient error. Then compute mean, sample standard deviation, and count over the three seeds without substituting values for non-converged runs.

- [ ] **Step 4: Generate figures and Chinese report**

Generate validation Top-1 vs optimizer steps, validation Top-1 vs wall time, loss curves, communication P50/P95 vs size, effective bandwidth, images/s, peak memory, convergence steps, and final accuracy. The report must state hardware/software hashes, sample counts, failures, mean±std, and distinguish observed results from interpretation.

- [ ] **Step 5: Verify report reproducibility**

Run aggregation and report generation twice into two clean directories, then compare SHA-256 of `summary.json`, `summary.csv`, and all figures. Expected: identical hashes.

- [ ] **Step 6: Download final artifacts to the local workspace**

Run locally:

```powershell
$runRoot = (ssh wangjun@192.168.1.100 'cat ~/work/ccdl_cifar10_runs/CURRENT_RUN').Trim()
$stamp = Split-Path $runRoot -Leaf
scp -r "wangjun@192.168.1.100:$runRoot/results" "G:\清华代码\ccdl-master\benchmark-results\cifar10-$stamp"
```

Expected: local raw logs, summary files, figures, checkpoints metadata, and Chinese report are present and readable.

### Task 7: Final verification gate

**Files:**
- Verify all files under the `benchmark-results/cifar10-$stamp` directory created by Task 6.

- [ ] **Step 1: Run local unit and syntax tests**

Run: `python -m pytest tests/benchmarks/cifar10 -q && python -m compileall -q benchmarks/cifar10`

Expected: all tests pass and compileall exits 0.

- [ ] **Step 2: Audit run completeness**

Check exactly 15 successful 200-epoch runs, three seeds per variant, 200 epoch records per run, finite mandatory metrics, and matching source manifest across all runs. Expected: no missing or duplicate run IDs.

- [ ] **Step 3: Audit claims against raw data**

Recompute at least one communication speedup, one final-accuracy delta, and one convergence-step delta directly from JSONL. Expected: exact agreement with report tables within displayed rounding.

- [ ] **Step 4: Record limitations**

Ensure the final report explicitly states that the controlled flat-buffer experiment does not measure DDP overlap, uses one two-GPU host, and does not yet generalize to language-model training.
