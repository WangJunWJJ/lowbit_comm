# CCDL Qwen2-0.5B Alpaca Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible two-GPU Qwen2-0.5B Alpaca benchmark comparing NCCL FP32 with four CCDL quantized gradient synchronization variants.

**Architecture:** A standalone `benchmarks/lm` package owns deterministic data preparation, Qwen2 language-weight loading, flat gradient synchronization, training/evaluation, matrix launch, aggregation, and reporting. It reuses CCDL public APIs but does not modify core CCDL behavior. Remote runs execute in the existing Transformers CUDA image and emit self-describing JSON artifacts.

**Tech Stack:** Python 3, PyTorch 2.4 distributed/NCCL, Transformers 4.44+, safetensors, CCDL CUDA extension, pytest, Docker, SVG/Markdown reporting.

## Global Constraints

- Hardware is exactly two RTX 4090 D GPUs on `wangjun@192.168.1.100`.
- Main variants are `nccl_fp32`, `int8_k0`, `int8_k2`, `int4_k0`, `int4_k2`; seeds are 17, 29, 43.
- Full-parameter BF16 training uses identical data order, batch/token budget, optimizer, scheduler, and evaluation cadence.
- Formal convergence is three consecutive evaluations at or below 1.01 times the mean final FP32 perplexity.
- No synthetic or smoke result may enter the formal summary.
- This workspace has no Git metadata, so commit steps are recorded as unavailable rather than fabricated.

---

### Task 1: Deterministic configuration and Alpaca tokenization

**Files:**
- Create: `benchmarks/lm/__init__.py`
- Create: `benchmarks/lm/config.py`
- Create: `benchmarks/lm/data.py`
- Test: `tests/benchmarks/lm/test_config.py`
- Test: `tests/benchmarks/lm/test_data.py`

**Interfaces:**
- Produces: `RunConfig`, `expand_main_matrix()`, `format_example()`, `split_indices()`, and response-only label construction.

- [ ] Write tests asserting 15 unique matrix entries, fixed variant parameters, deterministic disjoint 95/5 splits, and `-100` labels outside response tokens.
- [ ] Run `python -m pytest tests/benchmarks/lm/test_config.py tests/benchmarks/lm/test_data.py -q`; expect failures because modules do not exist.
- [ ] Implement immutable configuration, deterministic index splitting, prompt formatting, tokenization, and a collator that pads input IDs with the tokenizer pad ID and labels with `-100`.
- [ ] Re-run the focused tests; expect all pass.

### Task 2: Qwen2 text-model extraction and gradient synchronization

**Files:**
- Create: `benchmarks/lm/model.py`
- Create: `benchmarks/lm/sync.py`
- Test: `tests/benchmarks/lm/test_model.py`
- Test: `tests/benchmarks/lm/test_sync.py`

**Interfaces:**
- Consumes: `RunConfig`.
- Produces: `load_qwen2_text_model(path)`, `FlatGradientSynchronizer.synchronize(model) -> SyncMetrics`.

- [ ] Write CPU-safe tests for LLaVA-to-Qwen key remapping, tied embedding handling, flat-gradient copy-back, and variant validation.
- [ ] Run the tests and verify they fail before implementation.
- [ ] Implement config extraction from `text_config`, safetensors key filtering/remapping, strict missing/unexpected-key checks, and FP32 flat gradient synchronization using NCCL or CCDL.
- [ ] Re-run focused tests; expect all pass.

### Task 3: Training, evaluation, and structured logging

**Files:**
- Create: `benchmarks/lm/logging_utils.py`
- Create: `benchmarks/lm/train.py`
- Create: `benchmarks/lm/smoke.py`
- Test: `tests/benchmarks/lm/test_metrics.py`
- Test: `tests/benchmarks/lm/test_logging.py`

**Interfaces:**
- Consumes: model, tokenized datasets, synchronizer, `RunConfig`.
- Produces: per-rank environment metadata, rank-zero `metrics.jsonl`, `run.json`, checkpoints only when explicitly requested.

- [ ] Write tests for loss-to-perplexity overflow handling, response-token accuracy, weighted distributed metric reduction, JSONL append/read, and run completion markers.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement BF16 autocast, gradient accumulation with synchronization only at optimizer boundaries, AdamW and deterministic scheduler, validation, timing, throughput, peak-memory metrics, and failure recording.
- [ ] Re-run focused tests; expect all pass.

### Task 4: Aggregation, convergence, and report rendering

**Files:**
- Create: `benchmarks/lm/aggregate.py`
- Create: `benchmarks/lm/plot_report.py`
- Test: `tests/benchmarks/lm/test_aggregate.py`
- Test: `tests/benchmarks/lm/test_plot_report.py`

**Interfaces:**
- Consumes: 15 completed run directories plus preflight and environment files.
- Produces: `summary.json`, `summary.csv`, `runs.csv`, `CCDL_QWEN2_ALPACA_REPORT_ZH.md`, and SVG charts.

- [ ] Write tests for FP32-derived perplexity threshold, three-evaluation persistence, never-converged status, mean/std aggregation, and XML-escaped SVG labels.
- [ ] Verify tests fail, implement the minimal aggregation/report code, and re-run until pass.

### Task 5: Remote launcher and preflight closure

**Files:**
- Create: `benchmarks/lm/run_matrix.sh`
- Create: `benchmarks/lm/comm_bench.py`

**Interfaces:**
- Produces: timestamped run root with manifest, preflight records, microbenchmark, 15 isolated run directories, resumable completion detection, and `CURRENT_RUN` symlink.

- [ ] Implement a strict Bash launcher using the existing CUDA/Transformers image, `--gpus all`, host networking, `--shm-size=8g`, `NCCL_SOCKET_IFNAME=lo`, and `NCCL_IB_DISABLE=1`.
- [ ] Add checks for two visible GPUs, model/data hashes, language-weight load coverage, tokenizer round-trip, finite single-rank loss, two-rank synchronization error, and short runs of all five variants.
- [ ] Run all local benchmark tests and the existing CIFAR tests; expect all available tests pass.
- [ ] Copy the package to the remote worktree and run preflight. Record and diagnose any failure before formal launch.

### Task 6: Pilot budget calibration and full 15-run matrix

**Files:**
- Modify: `benchmarks/lm/run_matrix.sh` only if pilot evidence requires a common budget adjustment.
- Create remotely: `/home/wangjun/work/ccdl_lm_runs/<timestamp>/pilot/`
- Create remotely: `/home/wangjun/work/ccdl_lm_runs/<timestamp>/<variant>-seed<seed>/`

**Interfaces:**
- Consumes: passing preflight.
- Produces: complete real-GPU measurements with identical budgets.

- [ ] Run a fixed pilot for FP32 and the most lossy INT4 K0 variant; choose a common formal step count long enough to expose convergence without exceeding the agreed model/data scope.
- [ ] Freeze the manifest, launch all 15 configurations sequentially, and monitor epoch/step progress, GPU utilization, errors, disk, and completion markers.
- [ ] Restart only failed/incomplete configurations with the same manifest; never overwrite completed evidence.

### Task 7: Final verification and result handoff

**Files:**
- Create locally: `benchmark-results/qwen2-alpaca-<timestamp>/`

**Interfaces:**
- Consumes: completed matrix.
- Produces: verified local evidence and user-facing conclusions.

- [ ] Run `python -m benchmarks.lm.aggregate` and `python -m benchmarks.lm.plot_report` in the remote runtime.
- [ ] Verify exactly 15 successful formal runs, three seeds per variant, identical budgets, finite quality metrics, complete communication rows, and report/table consistency.
- [ ] Download the immutable run artifacts to local `benchmark-results/`, recompute hashes, and open the Markdown/SVG outputs for sanity review.
- [ ] Report communication speedup, end-to-end throughput, final perplexity/token accuracy, convergence step/time changes, wall-clock changes, Top-K trade-offs, limitations, and exact artifact links.
