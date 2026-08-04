# EF Policy and Async DDP Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CCDL Comm's DDP compression path policy-driven for error feedback and capable of a real async all-gather completion path.

**Architecture:** Keep the existing synchronous fused CUDA dequant-reduce path as the stable default. Add a small error-feedback policy layer between `CompressionConfig` and `ErrorFeedbackState`, then add an opt-in async all-gather transport that the DDP hook can use when supported.

**Tech Stack:** Python 3.10+, PyTorch distributed DDP communication hooks, CUDA/NCCL, pytest, existing `ccdl_cuda_ops` extension.

## Global Constraints

- Native PyTorch DDP gradient bucket compression on CUDA/NCCL is in scope.
- CANN/NPU async path is out of scope for this phase.
- FSDP integration is out of scope for this phase.
- New ring/tree/p2p algorithm migration is out of scope for this phase.
- Multi-node transport optimization is out of scope for this phase.
- Existing quantization format must not change.
- `async_gather` must default to `False`.
- Existing synchronous fused CUDA path must remain the stable fallback.
- Every implementation task must start with a failing test and end with a commit.

---

## File Structure

- Modify `ccdl_comm_refactor/ccdl_comm/config.py`
  - Owns user-facing compression configuration and validation.
- Create `ccdl_comm_refactor/ccdl_comm/quantization/error_feedback_policy.py`
  - Owns error-feedback scheduling decisions, bucket-local counters, and human-readable decision reasons.
- Modify `ccdl_comm_refactor/ccdl_comm/quantization/error_feedback.py`
  - Keep numerical residual storage only. Avoid adding policy logic here.
- Modify `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
  - Applies error-feedback policy and later chooses sync vs async all-gather path.
- Modify `ccdl_comm_refactor/ccdl_comm/communication/torch_transport.py`
  - Adds async same-size all-gather transport and fallback Future handling.
- Modify `ccdl_comm_refactor/ccdl_comm/collectives/work.py`
  - Adds a small async all-gather work protocol if needed by the transport.
- Modify `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
  - Adds benchmark flags for EF policy and async gather.
- Create or modify tests:
  - `ccdl_comm_refactor/tests/test_config.py`
  - `ccdl_comm_refactor/tests/test_error_feedback_policy.py`
  - `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`
  - `ccdl_comm_refactor/tests/test_torch_transport.py`
  - `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

---

### Task 1: Add Error Feedback Policy Configuration

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/config.py`
- Test: `ccdl_comm_refactor/tests/test_config.py`

**Interfaces:**
- Consumes: existing `CompressionConfig`.
- Produces:
  - `CompressionConfig.error_feedback_policy: str`
  - `CompressionConfig.error_feedback_min_numel: int`
  - `CompressionConfig.error_feedback_warmup_steps: int`
  - `CompressionConfig.error_feedback_period: int`
  - `CompressionConfig.effective_error_feedback_policy() -> str`

- [ ] **Step 1: Write failing config tests**

Append these tests to `ccdl_comm_refactor/tests/test_config.py`:

```python
import pytest

from ccdl_comm.config import CompressionConfig


def test_error_feedback_false_forces_none_policy() -> None:
    config = CompressionConfig(error_feedback=False, error_feedback_policy="always")

    assert config.effective_error_feedback_policy() == "none"


def test_error_feedback_true_uses_explicit_policy() -> None:
    config = CompressionConfig(error_feedback=True, error_feedback_policy="large_bucket_only")

    assert config.effective_error_feedback_policy() == "large_bucket_only"


def test_error_feedback_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported error_feedback_policy"):
        CompressionConfig(error_feedback_policy="sometimes")


def test_error_feedback_policy_rejects_negative_thresholds() -> None:
    with pytest.raises(ValueError, match="error_feedback_min_numel"):
        CompressionConfig(error_feedback_min_numel=-1)
    with pytest.raises(ValueError, match="error_feedback_warmup_steps"):
        CompressionConfig(error_feedback_warmup_steps=-1)
    with pytest.raises(ValueError, match="error_feedback_period"):
        CompressionConfig(error_feedback_period=0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_config.py -q
```

Expected: failures because the new `CompressionConfig` fields and `effective_error_feedback_policy` do not exist.

- [ ] **Step 3: Implement minimal config support**

In `ccdl_comm_refactor/ccdl_comm/config.py`, add:

```python
_SUPPORTED_ERROR_FEEDBACK_POLICIES = {
    "none",
    "always",
    "large_bucket_only",
    "warmup_then_enable",
    "periodic",
}
```

Extend the dataclass:

```python
    error_feedback_policy: str = "always"
    error_feedback_min_numel: int = 0
    error_feedback_warmup_steps: int = 0
    error_feedback_period: int = 1
```

Add validation in `__post_init__`:

```python
        if self.error_feedback_policy not in _SUPPORTED_ERROR_FEEDBACK_POLICIES:
            raise ValueError(
                "Unsupported error_feedback_policy="
                f"{self.error_feedback_policy!r}; expected one of {sorted(_SUPPORTED_ERROR_FEEDBACK_POLICIES)}"
            )
        if self.error_feedback_min_numel < 0:
            raise ValueError("error_feedback_min_numel must be >= 0")
        if self.error_feedback_warmup_steps < 0:
            raise ValueError("error_feedback_warmup_steps must be >= 0")
        if self.error_feedback_period <= 0:
            raise ValueError("error_feedback_period must be > 0")
```

Add method:

```python
    def effective_error_feedback_policy(self) -> str:
        if not self.error_feedback:
            return "none"
        return self.error_feedback_policy
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_config.py -q
```

Expected: all config tests pass.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/config.py ccdl_comm_refactor/tests/test_config.py
git commit -m "feat(ccdl_comm): configure error feedback policies"
```

---

### Task 2: Implement Error Feedback Policy Decisions

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/quantization/error_feedback_policy.py`
- Test: `ccdl_comm_refactor/tests/test_error_feedback_policy.py`

**Interfaces:**
- Consumes: `CompressionConfig.effective_error_feedback_policy() -> str`
- Produces:
  - `ErrorFeedbackDecision(apply: bool, update: bool, reason: str)`
  - `ErrorFeedbackPolicy(config: CompressionConfig)`
  - `ErrorFeedbackPolicy.decide(key: Hashable, *, numel: int) -> ErrorFeedbackDecision`
  - `ErrorFeedbackPolicy.advance(key: Hashable) -> None`

- [ ] **Step 1: Write failing policy tests**

Create `ccdl_comm_refactor/tests/test_error_feedback_policy.py`:

```python
from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.error_feedback_policy import ErrorFeedbackPolicy


def test_none_policy_never_applies_or_updates() -> None:
    policy = ErrorFeedbackPolicy(CompressionConfig(error_feedback=False))

    decision = policy.decide("bucket0", numel=10_000)

    assert not decision.apply
    assert not decision.update
    assert decision.reason == "error feedback disabled"


def test_always_policy_applies_and_updates() -> None:
    policy = ErrorFeedbackPolicy(CompressionConfig(error_feedback=True, error_feedback_policy="always"))

    decision = policy.decide("bucket0", numel=10_000)

    assert decision.apply
    assert decision.update
    assert decision.reason == "error feedback policy always"


def test_large_bucket_only_uses_numel_threshold() -> None:
    policy = ErrorFeedbackPolicy(
        CompressionConfig(
            error_feedback=True,
            error_feedback_policy="large_bucket_only",
            error_feedback_min_numel=4096,
        )
    )

    small = policy.decide("bucket0", numel=1024)
    large = policy.decide("bucket1", numel=4096)

    assert not small.apply
    assert not small.update
    assert small.reason == "bucket numel 1024 below error feedback threshold 4096"
    assert large.apply
    assert large.update
    assert large.reason == "bucket numel 4096 reached error feedback threshold 4096"


def test_warmup_then_enable_uses_bucket_local_steps() -> None:
    policy = ErrorFeedbackPolicy(
        CompressionConfig(
            error_feedback=True,
            error_feedback_policy="warmup_then_enable",
            error_feedback_warmup_steps=2,
        )
    )

    first = policy.decide("bucket0", numel=10_000)
    policy.advance("bucket0")
    second = policy.decide("bucket0", numel=10_000)
    policy.advance("bucket0")
    third = policy.decide("bucket0", numel=10_000)

    assert not first.apply
    assert first.reason == "bucket step 0 before error feedback warmup 2"
    assert not second.apply
    assert second.reason == "bucket step 1 before error feedback warmup 2"
    assert third.apply
    assert third.update
    assert third.reason == "bucket step 2 reached error feedback warmup 2"


def test_periodic_policy_applies_every_step_but_updates_periodically() -> None:
    policy = ErrorFeedbackPolicy(
        CompressionConfig(
            error_feedback=True,
            error_feedback_policy="periodic",
            error_feedback_period=3,
        )
    )

    decisions = []
    for _ in range(4):
        decisions.append(policy.decide("bucket0", numel=10_000))
        policy.advance("bucket0")

    assert [decision.apply for decision in decisions] == [True, True, True, True]
    assert [decision.update for decision in decisions] == [True, False, False, True]
    assert decisions[0].reason == "bucket step 0 updates error feedback every 3 steps"
    assert decisions[1].reason == "bucket step 1 skips error feedback update until period 3"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_error_feedback_policy.py -q
```

Expected: import failure because `error_feedback_policy.py` does not exist.

- [ ] **Step 3: Implement policy module**

Create `ccdl_comm_refactor/ccdl_comm/quantization/error_feedback_policy.py`:

```python
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from ccdl_comm.config import CompressionConfig


@dataclass(frozen=True)
class ErrorFeedbackDecision:
    apply: bool
    update: bool
    reason: str


@dataclass
class ErrorFeedbackPolicy:
    config: CompressionConfig
    _steps: dict[Hashable, int] = field(default_factory=dict)

    def decide(self, key: Hashable, *, numel: int) -> ErrorFeedbackDecision:
        policy = self.config.effective_error_feedback_policy()
        step = self._steps.get(key, 0)
        if policy == "none":
            return ErrorFeedbackDecision(False, False, "error feedback disabled")
        if policy == "always":
            return ErrorFeedbackDecision(True, True, "error feedback policy always")
        if policy == "large_bucket_only":
            threshold = self.config.error_feedback_min_numel
            if numel < threshold:
                return ErrorFeedbackDecision(
                    False,
                    False,
                    f"bucket numel {numel} below error feedback threshold {threshold}",
                )
            return ErrorFeedbackDecision(
                True,
                True,
                f"bucket numel {numel} reached error feedback threshold {threshold}",
            )
        if policy == "warmup_then_enable":
            warmup = self.config.error_feedback_warmup_steps
            if step < warmup:
                return ErrorFeedbackDecision(False, False, f"bucket step {step} before error feedback warmup {warmup}")
            return ErrorFeedbackDecision(True, True, f"bucket step {step} reached error feedback warmup {warmup}")
        if policy == "periodic":
            period = self.config.error_feedback_period
            should_update = step % period == 0
            if should_update:
                return ErrorFeedbackDecision(True, True, f"bucket step {step} updates error feedback every {period} steps")
            return ErrorFeedbackDecision(True, False, f"bucket step {step} skips error feedback update until period {period}")
        raise ValueError(f"unsupported error feedback policy: {policy}")

    def advance(self, key: Hashable) -> None:
        self._steps[key] = self._steps.get(key, 0) + 1
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_error_feedback_policy.py ccdl_comm_refactor/tests/test_config.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/quantization/error_feedback_policy.py ccdl_comm_refactor/tests/test_error_feedback_policy.py
git commit -m "feat(ccdl_comm): add error feedback policy decisions"
```

---

### Task 3: Apply Error Feedback Policy in DDP Hook

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Test: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`

**Interfaces:**
- Consumes:
  - `ErrorFeedbackPolicy.decide(key, numel=...)`
  - `ErrorFeedbackPolicy.advance(key)`
- Produces:
  - DDP hook applies `feedback.compensate` only when `decision.apply`.
  - DDP hook applies `feedback.update` only when `decision.update`.
  - DDP hook advances policy once per processed compressed bucket.

- [ ] **Step 1: Write failing DDP hook tests**

Append tests to `ccdl_comm_refactor/tests/test_ddp_comm_hook.py` using existing `FakeTensor`, `FakeBucket`, and `FakeFuture` helpers:

```python
def test_all_gather_hook_skips_error_feedback_for_small_bucket_policy(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer="rank0", shape=(2,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(2,), dtype="fp16"),
            ],
            world_size=2,
        )

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", kwargs["reduce"]))
        return FakeTensor([2.0, 4.0])

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key))
            return FakeTensor([10.0, 20.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key))

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(
            bit=8,
            error_feedback=True,
            error_feedback_policy="large_bucket_only",
            error_feedback_min_numel=4,
        ),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        all_gather=all_gather,
        error_feedback=Feedback(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([2.0, 4.0])
    assert ("compensate", 0) not in calls
    assert ("update", 0) not in calls
    assert ("quantize", FakeTensor([1.0, 2.0])) in calls


def test_all_gather_hook_updates_error_feedback_when_policy_allows(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
            ],
            world_size=2,
        )

    def dequantize_reduce(buffers, shape, config, **kwargs):
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([10.0, 20.0, 30.0, 40.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key, original, transmitted))

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(
            bit=8,
            error_feedback=True,
            error_feedback_policy="large_bucket_only",
            error_feedback_min_numel=4,
        ),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        all_gather=all_gather,
        error_feedback=Feedback(),
        future_factory=FakeFuture,
    )

    hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert ("compensate", 0, FakeTensor([1.0, 2.0, 3.0, 4.0])) in calls
    assert (
        "update",
        0,
        FakeTensor([10.0, 20.0, 30.0, 40.0]),
        FakeTensor([2.0, 4.0, 6.0, 8.0]),
    ) in calls
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py::test_all_gather_hook_skips_error_feedback_for_small_bucket_policy ccdl_comm_refactor/tests/test_ddp_comm_hook.py::test_all_gather_hook_updates_error_feedback_when_policy_allows -q
```

Expected: first test fails because current hook uses boolean `config.error_feedback` and always compensates/updates.

- [ ] **Step 3: Implement policy integration**

In `ddp_hook.py`, import:

```python
from ccdl_comm.quantization.error_feedback_policy import ErrorFeedbackPolicy
```

After `feedback = error_feedback or ErrorFeedbackState()`, add:

```python
    feedback_policy = ErrorFeedbackPolicy(config)
```

In the all-gather `process_bucket`, replace:

```python
            prepared = feedback.compensate(key, original) if config.error_feedback else original
```

with:

```python
            feedback_decision = feedback_policy.decide(key, numel=_numel(original))
            prepared = feedback.compensate(key, original) if feedback_decision.apply else original
```

Replace each:

```python
                if config.error_feedback:
                    feedback.update(key, original=prepared, transmitted=restored)
```

with:

```python
                if feedback_decision.update:
                    feedback.update(key, original=prepared, transmitted=restored)
                feedback_policy.advance(key)
```

Make the same policy decision in the `all_reduce` strategy by passing a policy-aware feedback wrapper or by keeping `DDPBucketProcessor` unchanged and routing policy integration only through the all-gather path in this task. If choosing the latter, document in code comments that policy integration for `strategy="all_reduce"` remains unchanged because ParaScale's validated path is `all_gather`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_error_feedback_policy.py -q
```

Expected: pass.

- [ ] **Step 5: Run full local refactor tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests -q -rs
```

Expected: pass, with skips only for unavailable local torch/CANN dependencies.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py
git commit -m "perf(ccdl_comm): apply error feedback bucket policy"
```

---

### Task 4: Add Benchmark Flags for Error Feedback Policies

**Files:**
- Modify: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
- Test: `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

**Interfaces:**
- Consumes: `CompressionConfig` EF policy fields.
- Produces CLI flags:
  - `--error-feedback {true,false}`
  - `--error-feedback-policy {none,always,large_bucket_only,warmup_then_enable,periodic}`
  - `--error-feedback-min-numel INT`
  - `--error-feedback-warmup-steps INT`
  - `--error-feedback-period INT`

- [ ] **Step 1: Write failing script tests**

Append to `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`:

```python
from pathlib import Path


def test_synthetic_ddp_script_exposes_error_feedback_policy_flags() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--error-feedback" in source
    assert "--error-feedback-policy" in source
    assert "--error-feedback-min-numel" in source
    assert "--error-feedback-warmup-steps" in source
    assert "--error-feedback-period" in source
    assert "error_feedback_policy=args.error_feedback_policy" in source
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_synthetic_ddp_script.py::test_synthetic_ddp_script_exposes_error_feedback_policy_flags -q
```

Expected: fail because flags are missing.

- [ ] **Step 3: Implement script flags**

In `parse_args()`, add:

```python
    parser.add_argument("--error-feedback", choices=("true", "false"), default="true")
    parser.add_argument(
        "--error-feedback-policy",
        choices=("none", "always", "large_bucket_only", "warmup_then_enable", "periodic"),
        default="always",
    )
    parser.add_argument("--error-feedback-min-numel", type=int, default=0)
    parser.add_argument("--error-feedback-warmup-steps", type=int, default=0)
    parser.add_argument("--error-feedback-period", type=int, default=1)
```

In `build_model`, replace `CompressionConfig(bit=args.bit, group_size=args.group_size, error_feedback=True)` with:

```python
                CompressionConfig(
                    bit=args.bit,
                    group_size=args.group_size,
                    error_feedback=(args.error_feedback == "true"),
                    error_feedback_policy=args.error_feedback_policy,
                    error_feedback_min_numel=args.error_feedback_min_numel,
                    error_feedback_warmup_steps=args.error_feedback_warmup_steps,
                    error_feedback_period=args.error_feedback_period,
                ),
```

In result JSON, add:

```python
            "error_feedback": args.error_feedback if args.mode == "ccdl" else None,
            "error_feedback_policy": args.error_feedback_policy if args.mode == "ccdl" else None,
            "error_feedback_min_numel": args.error_feedback_min_numel if args.mode == "ccdl" else None,
            "error_feedback_warmup_steps": args.error_feedback_warmup_steps if args.mode == "ccdl" else None,
            "error_feedback_period": args.error_feedback_period if args.mode == "ccdl" else None,
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_synthetic_ddp_script.py ccdl_comm_refactor/tests/test_config.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py
git commit -m "test(ccdl_comm): expose error feedback benchmark policies"
```

---

### Task 5: Add Async All-Gather Transport

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/torch_transport.py`
- Test: `ccdl_comm_refactor/tests/test_torch_transport.py`

**Interfaces:**
- Produces:
  - `AsyncGatheredPayloads(payloads: Sequence[Any], world_size: int)`
  - `AsyncAllGatherWork.wait() -> GatheredPayloads`
  - `AsyncAllGatherWork.get_future() -> Any`
  - `make_torch_async_all_gather(import_module=...) -> Callable[[Any], AsyncAllGatherWork]`

- [ ] **Step 1: Write failing async transport tests**

Append to `ccdl_comm_refactor/tests/test_torch_transport.py`:

```python
def test_async_all_gather_transport_returns_work_with_future() -> None:
    calls = []

    class FakeFuture:
        def __init__(self):
            self.callbacks = []

        def then(self, callback):
            self.callbacks.append(callback)
            return self

    class FakeHandle:
        def __init__(self):
            self.future = FakeFuture()

        def wait(self):
            calls.append("wait")

        def get_future(self):
            calls.append("get_future")
            return self.future

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather(output_list, buffer, async_op=False):
            calls.append(("all_gather", len(output_list), async_op))
            output_list[0] = "rank0"
            output_list[1] = "rank1"
            return FakeHandle()

    class FakeBuffer:
        shape = (3,)

        def new_empty(self, shape):
            return ("empty", shape)

    def import_module(name):
        assert name == "torch.distributed"
        return FakeDist

    transport = make_torch_async_all_gather(import_module=import_module)
    work = transport(FakeBuffer())

    assert work.get_future() is work.handle.future
    assert work.wait().payloads == ["rank0", "rank1"]
    assert calls == [("all_gather", 2, True), "get_future", "wait"]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_torch_transport.py::test_async_all_gather_transport_returns_work_with_future -q
```

Expected: fail because `make_torch_async_all_gather` does not exist.

- [ ] **Step 3: Implement async transport**

In `torch_transport.py`, import:

```python
from dataclasses import dataclass
from collections.abc import Sequence
```

Add:

```python
@dataclass
class AsyncAllGatherWork:
    payloads: Sequence[Any]
    world_size: int
    handle: Any

    def wait(self) -> GatheredPayloads:
        self.handle.wait()
        return GatheredPayloads(payloads=self.payloads, world_size=self.world_size)

    def get_future(self) -> Any:
        get_future = getattr(self.handle, "get_future", None)
        if callable(get_future):
            return get_future()
        return None
```

Add:

```python
def make_torch_async_all_gather(
    *,
    import_module: Callable[[str], Any] = _import_module,
) -> Callable[[Any], AsyncAllGatherWork]:
    """Create an async same-size all-gather transport backed by torch.distributed."""

    def transport(buffer: Any) -> AsyncAllGatherWork:
        try:
            dist = import_module("torch.distributed")
        except (ImportError, ModuleNotFoundError) as exc:
            raise TorchDistributedUnavailableError("torch.distributed is not available") from exc

        if not dist.is_available() or not dist.is_initialized():
            raise TorchDistributedUnavailableError("torch.distributed is not initialized")

        world_size = dist.get_world_size()
        output_shape = tuple(getattr(buffer, "shape", ()))
        output_list = [buffer.new_empty(output_shape) for _ in range(world_size)]
        handle = dist.all_gather(output_list, buffer, async_op=True)
        return AsyncAllGatherWork(payloads=output_list, world_size=world_size, handle=handle)

    return transport
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_torch_transport.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/torch_transport.py ccdl_comm_refactor/tests/test_torch_transport.py
git commit -m "feat(ccdl_comm): add async all gather transport"
```

---

### Task 6: Integrate Async All-Gather into DDP Hook

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Test: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`

**Interfaces:**
- Consumes: `make_torch_async_all_gather()`, `AsyncAllGatherWork.wait()`, `AsyncAllGatherWork.get_future()`.
- Produces:
  - `create_ddp_comm_hook(..., async_gather: bool = False, async_all_gather: Callable[[Any], Any] | None = None)`
  - Async path only for `strategy="all_gather"` and default dequantize fastpath.

- [ ] **Step 1: Write failing DDP async hook test**

Append to `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`:

```python
def test_all_gather_hook_can_complete_from_async_gather_future(monkeypatch) -> None:
    calls = []

    class FakeTorchFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class FakeGatherWork:
        def __init__(self):
            self.payloads = [
                CompressedPayload(buffer="rank0", shape=(2,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(2,), dtype="fp16"),
            ]
            self.world_size = 2

        def get_future(self):
            calls.append("get_future")
            return FakeTorchFuture()

        def wait(self):
            calls.append("wait")
            return GatheredPayloads(payloads=self.payloads, world_size=self.world_size)

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        calls.append(("async_all_gather", buffer))
        return FakeGatherWork()

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", buffers, kwargs["reduce"]))
        return FakeTensor([2.0, 4.0])

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_all_gather=async_all_gather,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([2.0, 4.0])
    assert calls == [
        ("quantize", FakeTensor([1.0, 2.0])),
        ("async_all_gather", "local-buffer"),
        "get_future",
        "then",
        "wait",
        ("dequantize_reduce", ["rank0", "rank1"], "mean"),
    ]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py::test_all_gather_hook_can_complete_from_async_gather_future -q
```

Expected: fail because `create_ddp_comm_hook` has no `async_gather` parameter.

- [ ] **Step 3: Implement async hook path**

In `ddp_hook.py`, import:

```python
from ccdl_comm.communication.torch_transport import make_torch_async_all_gather
```

Extend `create_ddp_comm_hook` signature:

```python
    async_gather: bool = False,
    async_all_gather: Callable[[Any], Any] | None = None,
```

Inside `strategy == "all_gather"` setup, create:

```python
        active_async_all_gather = async_all_gather or make_torch_async_all_gather()
```

In `process_bucket`, before the synchronous `active_all_gather(local_payload)` path, add an async branch for default fastpath:

```python
                if async_gather and dequantize is None and reduce in {"mean", "sum"}:
                    local_payload = _coerce_payload(
                        active_quantize(prepared, config),
                        shape=tuple(prepared.shape),
                        dtype=active_dtype,
                    )
                    gather_work = active_async_all_gather(_payload_buffer(local_payload))
                    outer_future = future_factory()

                    def complete(_ignored: Any = None) -> Any:
                        gathered = gather_work.wait()
                        restored = dequantize_reduce_tensors(
                            [_payload_buffer(payload) for payload in gathered.payloads],
                            tuple(prepared.shape),
                            config,
                            dtype=active_dtype,
                            extension_status=extension_status,
                            reduce=reduce,
                        )
                        if feedback_decision.update:
                            feedback.update(key, original=prepared, transmitted=restored)
                        feedback_policy.advance(key)
                        outer_future.set_result(restored)
                        return restored

                    inner_future = gather_work.get_future()
                    if inner_future is not None and hasattr(inner_future, "then"):
                        inner_future.then(complete)
                    else:
                        complete()
                    return outer_future
```

Because `process_bucket` can now return a Future directly, change the outer `hook`:

```python
    def hook(state: Any, bucket: Any) -> Any:
        result = process_bucket(bucket)
        if hasattr(result, "set_result"):
            return result
        future = future_factory()
        future.set_result(result)
        return future
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_torch_transport.py -q
```

Expected: pass.

- [ ] **Step 5: Run full local tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests -q -rs
```

Expected: pass, with skips only for unavailable local torch/CANN dependencies.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py
git commit -m "feat(ccdl_comm): enable async all gather ddp hook"
```

---

### Task 7: Add Async Benchmark Flags and Remote Validation

**Files:**
- Modify: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
- Modify: `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

**Interfaces:**
- Consumes: `create_ddp_comm_hook(async_gather=...)`.
- Produces CLI flag:
  - `--async-gather {true,false}`

- [ ] **Step 1: Write failing script test**

Append to `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`:

```python
def test_synthetic_ddp_script_exposes_async_gather_flag() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--async-gather" in source
    assert "async_gather=(args.async_gather == \"true\")" in source
    assert '"async_gather": args.async_gather if args.mode == "ccdl" else None' in source
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_synthetic_ddp_script.py::test_synthetic_ddp_script_exposes_async_gather_flag -q
```

Expected: fail because flag is absent.

- [ ] **Step 3: Implement benchmark flag**

In `parse_args()`, add:

```python
    parser.add_argument("--async-gather", choices=("true", "false"), default="false")
```

In `create_ddp_comm_hook(...)`, pass:

```python
                async_gather=(args.async_gather == "true"),
```

In result JSON, add:

```python
            "async_gather": args.async_gather if args.mode == "ccdl" else None,
```

- [ ] **Step 4: Run local tests**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_synthetic_ddp_script.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py
git commit -m "test(ccdl_comm): benchmark async ddp gather"
```

- [ ] **Step 6: Remote validation**

On the A6000 server, rebuild and test:

```bash
cd /home/user/wangjun/ccdl-master/ccdl_comm_refactor
export CCDL_COMM_BUILD_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.6
rm -rf build ccdl_cuda_ops*.so
python setup.py build_ext --inplace
python -m pytest tests -q -rs --tb=short
```

Then run 2-GPU and 4-GPU benchmarks:

```bash
COMMON="--steps 50 --warmup-steps 10 --batch-size-per-rank 8 --input-dim 2048 --width 4096 --depth 3 --output-dim 1024 --bucket-cap-mb 25 --model-dtype fp32 --lr 0.0001"

torchrun --standalone --nproc_per_node=2 tests/distributed/synthetic_ddp_compare.py --mode ccdl --output-json /work/results/ef_async/2gpu_large_bucket_ef.json $COMMON --bit 8 --group-size 64 --strategy all_gather --min-compress-numel 0 --error-feedback true --error-feedback-policy large_bucket_only --error-feedback-min-numel 4000000
torchrun --standalone --nproc_per_node=2 tests/distributed/synthetic_ddp_compare.py --mode ccdl --output-json /work/results/ef_async/2gpu_async_no_ef.json $COMMON --bit 8 --group-size 64 --strategy all_gather --min-compress-numel 0 --error-feedback false --async-gather true
torchrun --standalone --nproc_per_node=4 tests/distributed/synthetic_ddp_compare.py --mode ccdl --output-json /work/results/ef_async/4gpu_large_bucket_ef.json $COMMON --bit 8 --group-size 64 --strategy all_gather --min-compress-numel 0 --error-feedback true --error-feedback-policy large_bucket_only --error-feedback-min-numel 4000000
torchrun --standalone --nproc_per_node=4 tests/distributed/synthetic_ddp_compare.py --mode ccdl --output-json /work/results/ef_async/4gpu_async_no_ef.json $COMMON --bit 8 --group-size 64 --strategy all_gather --min-compress-numel 0 --error-feedback false --async-gather true
```

Expected:

- Remote tests pass.
- Async path is correct.
- Report whether async path improves, matches, or regresses relative to synchronous path.

---

## Self-Review

- Spec coverage:
  - EF policy config: Task 1.
  - EF policy runtime decisions: Task 2.
  - DDP hook policy integration: Task 3.
  - Benchmark policy flags: Task 4.
  - Async transport: Task 5.
  - Async DDP hook: Task 6.
  - Remote benchmark validation: Task 7.
- Placeholder scan:
  - No unresolved planning markers are intentionally left.
- Type consistency:
  - `ErrorFeedbackDecision`, `ErrorFeedbackPolicy.decide`, `ErrorFeedbackPolicy.advance`, and `async_gather` names are consistent across tasks.
