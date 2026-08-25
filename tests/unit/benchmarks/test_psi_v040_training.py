"""Contracts for the paired v0.4.0 PSI three-route training matrix."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from inspect import getsource
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.benchmarks.psi_v040_state import ShardLayout, ShardedAdamW
from tests.benchmarks.psi_v040_training import (
    PAIRED_SEEDS,
    ROUTES,
    PairedRouteFacts,
    ResumeFacts,
    RSAGQWDTransaction,
    StepTiming,
    assert_paired_route_parity,
    assert_resume_matches,
    build_engine,
    build_step_record,
    build_task_result,
    parse_args,
    validate_step_record,
    validate_task_result,
)
from tests.benchmarks.distributed_psi_v040_worker import (
    CAGUpdateEngine,
    NativeUpdateEngine,
    RSAGQWDUpdateEngine,
    _HookTelemetry,
    _advance_amp_scaler,
    _build_workspace,
    _install_psi_update_seams,
    _load_checkpoint,
    _load_resume_oracle,
    _register_ddp_hook,
    _reject_locked_psi_overrides,
    _resolve_resume_path,
    _run,
    _stable_ddp_bucket_cap_mb,
    _validate_epoch,
    _validate_amp_configuration,
    _write_raw_records,
    _write_resume_oracle,
    source_tree_manifest,
    summarize_step_records,
)


def _parity_facts() -> PairedRouteFacts:
    return PairedRouteFacts(
        initial_parameter_sha256="a" * 64,
        sampler_indices=(9, 2, 7, 1),
        augmentation_rng_sha256="b" * 64,
        lr_schedule=(0.0, 1.0e-4, 9.0e-5),
        amp_configuration=(
            "fp16",
            True,
            65536.0,
            65536.0,
            2000,
            2.0,
            0.5,
        ),
        batch_size=16,
        model_parameter_count=44_956_000,
    )


def _step_record() -> dict[str, object]:
    return build_step_record(
        task_id="20260821-native",
        attempt_id="attempt-1",
        route="native",
        seed=20260821,
        epoch=0,
        step=1,
        batch_indices=(4, 5),
        timing=StepTiming(
            forward_s=1.0,
            backward_s=2.0,
            update_s=3.0,
            communication_s=4.0,
            validation_s=100.0,
            report_serialization_s=200.0,
        ),
        gradient_route="ddp_nccl",
        parameter_route="full_adamw",
        communication_bytes=128,
        qwd_s=0.0,
        refresh_s=0.0,
        decision="native",
        loss=0.5,
        amp_scale=65536.0,
        learning_rate=1.0e-4,
        model_sha256="c" * 64,
        rank_parameter_gap=0.0,
        optimizer_step=1,
        finite=True,
    )


def _task_result() -> dict[str, object]:
    return build_task_result(
        task_id="20260821-native",
        attempt_id="attempt-1",
        route="native",
        seed=20260821,
        world_size=4,
        physical_gpu_ids=(1, 2, 3, 4),
        source_manifest_sha256="d" * 64,
        data_sha256="e" * 64,
        parity=_parity_facts(),
        epochs=3,
        steps=3,
        warmup_steps=1,
        steady_samples_per_second=100.0,
        step_latency_p50_ms=600.0,
        step_latency_p95_ms=650.0,
        epoch_time_s=(1300.0, 1290.0, 1280.0),
        communication_time_s=40.0,
        qwd_time_s=0.0,
        refresh_time_s=0.0,
        communication_bytes=1234,
        peak_memory_mib=23456.0,
        gpu_telemetry=tuple(
            {
                "gpu": gpu,
                "utilization": 91.0,
                "memory_used_mib": 1234.0,
                "temperature_c": 42.0,
                "sm_clock_mhz": 1905.0,
            }
            for gpu in (1, 2, 3, 4)
        ),
        loss_trajectory=(0.9, 0.7, 0.5),
        validation_loss=0.4,
        rank_gaps=(0.0, 0.0, 0.0),
        decision_counts={"native": 3},
        failure_facts=(),
    )


def test_matrix_exposes_only_the_approved_routes_and_seeds() -> None:
    assert ROUTES == ("native", "cag", "rsag_qwd")
    assert PAIRED_SEEDS == (20260821, 20260822, 20260823)


def test_cli_rejects_unknown_route() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--route", "rsag"])


def test_cli_accepts_each_exact_route() -> None:
    for route in ROUTES:
        args = parse_args(["--route", route])
        assert args.route == route


def test_cli_carries_checkpoint_data_and_repeatable_psi_overrides() -> None:
    args = parse_args(
        [
            "--route",
            "native",
            "--resume",
            "/tmp/checkpoint.pt",
            "--checkpoint-dir",
            "/tmp/checkpoints",
            "--data-sha256",
            "a" * 64,
            "--amp-initial-scale",
            "1024",
            "--amp-growth-interval",
            "1",
            "--psi-override",
            "cache=none",
            "--psi-override",
            "training.torch_compile.enabled=false",
        ]
    )

    assert args.resume == "/tmp/checkpoint.pt"
    assert args.checkpoint_dir == "/tmp/checkpoints"
    assert args.data_sha256 == "a" * 64
    assert args.amp_initial_scale == 1024.0
    assert args.amp_growth_interval == 1
    assert args.psi_override == [
        "cache=none",
        "training.torch_compile.enabled=false",
    ]


def test_cli_preserves_explicit_cuda_process_placement_values() -> None:
    affinity_map = "0-9,40-49;10-19,50-59;20-29,60-69;30-39,70-79"
    args = parse_args(
        [
            "--route",
            "rsag_qwd",
            "--cpu-affinity-map",
            affinity_map,
            "--nccl-channels",
            "4",
        ]
    )

    assert args.cpu_affinity_map == affinity_map
    assert args.nccl_channels == 4


def test_worker_delegates_placement_before_training() -> None:
    module = sys.modules["tests.benchmarks.distributed_psi_v040_worker"]
    source = getsource(module)
    main_source = getsource(module.main)

    assert "def _apply_cpu_affinity" not in source
    assert "def _apply_nccl_channels" not in source
    assert "parse_cuda_process_placement" in source
    assert "apply_cuda_process_placement" in source
    assert main_source.index("parse_cuda_process_placement") < (
        main_source.index("apply_cuda_process_placement")
    )
    assert main_source.index("apply_cuda_process_placement") < (
        main_source.index("_run(args)")
    )


def test_resume_path_expands_one_exact_rank_placeholder() -> None:
    assert _resolve_resume_path("/tmp/midpoint-rank{rank}.pt", 3) == Path(
        "/tmp/midpoint-rank3.pt"
    )

    with pytest.raises(ValueError, match="placeholder"):
        _resolve_resume_path("/tmp/{seed}.pt", 0)


def test_all_routes_must_receive_identical_paired_inputs() -> None:
    facts = _parity_facts()

    assert_paired_route_parity({route: facts for route in ROUTES})


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("initial_parameter_sha256", "f" * 64),
        ("sampler_indices", (9, 2, 1, 7)),
        ("augmentation_rng_sha256", "f" * 64),
        ("lr_schedule", (0.0, 8.0e-5, 7.0e-5)),
        (
            "amp_configuration",
            ("fp16", False, 65536.0, 65536.0, 2000, 2.0, 0.5),
        ),
        ("batch_size", 8),
        ("model_parameter_count", 1),
    ],
)
def test_pairing_rejects_each_route_parity_drift(
    field: str,
    replacement: object,
) -> None:
    native = _parity_facts()
    changed = {name: getattr(native, name) for name in native.__slots__}
    changed[field] = replacement

    with pytest.raises(ValueError, match=field):
        assert_paired_route_parity(
            {
                "native": native,
                "cag": PairedRouteFacts(**changed),
                "rsag_qwd": native,
            }
        )


def test_pairing_requires_one_fact_set_for_every_exact_route() -> None:
    with pytest.raises(ValueError, match="route fields"):
        assert_paired_route_parity({"native": _parity_facts()})


def test_performance_window_includes_only_training_phases() -> None:
    timing = StepTiming(
        forward_s=1.0,
        backward_s=2.0,
        update_s=3.0,
        communication_s=4.0,
        validation_s=100.0,
        report_serialization_s=200.0,
    )

    assert timing.measured_s == 10.0
    assert timing.total_wall_s == 310.0
    assert timing.to_dict() == {
        "forward_s": 1.0,
        "backward_s": 2.0,
        "update_s": 3.0,
        "communication_s": 4.0,
        "validation_s": 100.0,
        "report_serialization_s": 200.0,
        "measured_s": 10.0,
    }


def test_step_schema_is_exact_and_separates_timing_domains() -> None:
    record = _step_record()

    assert validate_step_record(record) == record
    assert set(record) == {
        "schema_version",
        "task_id",
        "attempt_id",
        "route",
        "seed",
        "epoch",
        "step",
        "batch_indices",
        "timing",
        "communication",
        "quality",
    }
    assert set(record["timing"]) == {
        "forward_s",
        "backward_s",
        "update_s",
        "communication_s",
        "validation_s",
        "report_serialization_s",
        "measured_s",
    }
    assert set(record["communication"]) == {
        "gradient_route",
        "parameter_route",
        "bytes",
        "qwd_s",
        "refresh_s",
        "decision",
    }
    assert set(record["quality"]) == {
        "loss",
        "amp_scale",
        "learning_rate",
        "model_sha256",
        "rank_parameter_gap",
        "optimizer_step",
        "finite",
    }


@pytest.mark.parametrize(
    "container", ("top", "timing", "communication", "quality")
)
def test_step_schema_rejects_extra_fields(container: str) -> None:
    record = deepcopy(_step_record())
    target = record if container == "top" else record[container]
    target["unexpected"] = True

    with pytest.raises(ValueError, match="fields"):
        validate_step_record(record)


def test_task_result_schema_is_exact_and_has_all_release_metrics() -> None:
    result = _task_result()

    assert validate_task_result(result) == result
    assert set(result) == {
        "schema_version",
        "task_id",
        "attempt_id",
        "route",
        "seed",
        "world_size",
        "physical_gpu_ids",
        "source_manifest_sha256",
        "data_sha256",
        "initial_parameter_sha256",
        "sampler_indices_sha256",
        "augmentation_rng_sha256",
        "lr_schedule_sha256",
        "amp_configuration",
        "batch_size_per_rank",
        "global_batch_size",
        "model_parameter_count",
        "epochs",
        "steps",
        "warmup_steps",
        "steady_samples_per_second",
        "step_latency_p50_ms",
        "step_latency_p95_ms",
        "epoch_time_s",
        "communication_time_s",
        "qwd_time_s",
        "refresh_time_s",
        "communication_bytes",
        "peak_memory_mib",
        "gpu_telemetry",
        "loss_trajectory",
        "validation_loss",
        "rank_gaps",
        "decision_counts",
        "failure_facts",
    }


def test_task_result_schema_rejects_missing_extra_and_unknown_route() -> None:
    result = _task_result()
    missing = deepcopy(result)
    del missing["rank_gaps"]
    extra = deepcopy(result)
    extra["notes"] = "not schema-v1"
    wrong_route = deepcopy(result)
    wrong_route["route"] = "rsag"

    for invalid in (missing, extra, wrong_route):
        with pytest.raises(ValueError):
            validate_task_result(invalid)


def test_resume_facts_match_next_batch_lr_amp_optimizer_model_and_loss() -> (
    None
):
    oracle = ResumeFacts(
        next_batch_indices=(20, 21, 22),
        next_batch_sha256="a" * 64,
        next_augmentation_sha256="b" * 64,
        learning_rate=8.0e-5,
        amp_scale=32768.0,
        optimizer_state_sha256="1" * 64,
        model_sha256="2" * 64,
        next_loss=0.125,
        post_learning_rate=9.0e-5,
        post_amp_scale=32768.0,
        post_optimizer_state_sha256="3" * 64,
        post_model_sha256="4" * 64,
    )

    assert_resume_matches(oracle, oracle)


@pytest.mark.parametrize(
    "field",
    (
        "next_batch_indices",
        "next_batch_sha256",
        "next_augmentation_sha256",
        "learning_rate",
        "amp_scale",
        "optimizer_state_sha256",
        "model_sha256",
        "next_loss",
        "post_learning_rate",
        "post_amp_scale",
        "post_optimizer_state_sha256",
        "post_model_sha256",
    ),
)
def test_resume_comparison_rejects_every_required_drift(field: str) -> None:
    oracle = ResumeFacts(
        next_batch_indices=(20, 21, 22),
        next_batch_sha256="a" * 64,
        next_augmentation_sha256="b" * 64,
        learning_rate=8.0e-5,
        amp_scale=32768.0,
        optimizer_state_sha256="1" * 64,
        model_sha256="2" * 64,
        next_loss=0.125,
        post_learning_rate=9.0e-5,
        post_amp_scale=32768.0,
        post_optimizer_state_sha256="3" * 64,
        post_model_sha256="4" * 64,
    )
    changed = {name: getattr(oracle, name) for name in oracle.__slots__}
    changed[field] = {
        "next_batch_indices": (99,),
        "next_batch_sha256": "7" * 64,
        "next_augmentation_sha256": "8" * 64,
        "learning_rate": 7.0e-5,
        "amp_scale": 1.0,
        "optimizer_state_sha256": "3" * 64,
        "model_sha256": "4" * 64,
        "next_loss": 0.25,
        "post_learning_rate": 7.0e-5,
        "post_amp_scale": 1.0,
        "post_optimizer_state_sha256": "5" * 64,
        "post_model_sha256": "6" * 64,
    }[field]

    with pytest.raises(ValueError, match=field):
        assert_resume_matches(oracle, ResumeFacts(**changed))


class _Work:
    def __init__(
        self, result: object = None, failure: Exception | None = None
    ):
        self._result = result
        self._failure = failure

    def wait(self) -> object:
        if self._failure is not None:
            raise self._failure
        return self._result


def test_rsag_qwd_publishes_optimizer_then_model_only_after_work_success() -> (
    None
):
    events: list[tuple[str, object]] = []
    transaction = RSAGQWDTransaction(
        publish_optimizer=lambda state: events.append(("optimizer", state)),
        publish_model=lambda model: events.append(("model", model)),
    )

    transaction.commit("candidate-state", _Work("candidate-model"))

    assert events == [
        ("optimizer", "candidate-state"),
        ("model", "candidate-model"),
    ]


def test_rsag_qwd_failure_publishes_neither_optimizer_nor_model() -> None:
    events: list[tuple[str, object]] = []
    transaction = RSAGQWDTransaction(
        publish_optimizer=lambda state: events.append(("optimizer", state)),
        publish_model=lambda model: events.append(("model", model)),
    )

    with pytest.raises(RuntimeError, match="collective failed"):
        transaction.commit(
            "candidate-state",
            _Work(failure=RuntimeError("collective failed")),
        )

    assert events == []


def test_route_factory_dispatches_to_one_exact_engine() -> None:
    events: list[str] = []

    for expected in ROUTES:
        engine = build_engine(
            expected,
            native_factory=lambda: events.append("native") or "native-engine",
            cag_factory=lambda: events.append("cag") or "cag-engine",
            rsag_qwd_factory=lambda: (
                events.append("rsag_qwd") or "rsag-engine"
            ),
        )
        assert engine == (
            "rsag-engine" if expected == "rsag_qwd" else f"{expected}-engine"
        )

    assert events == list(ROUTES)


def test_route_factory_rejects_unknown_route_before_calling_factories() -> (
    None
):
    def forbidden() -> object:
        raise AssertionError("factory must not run")

    with pytest.raises(ValueError, match="route"):
        build_engine(
            "rsag",
            native_factory=forbidden,
            cag_factory=forbidden,
            rsag_qwd_factory=forbidden,
        )


def test_worker_module_is_cpu_importable_and_names_exact_engines() -> None:
    assert NativeUpdateEngine.route == "native"
    assert CAGUpdateEngine.route == "cag"
    assert RSAGQWDUpdateEngine.route == "rsag_qwd"


def test_rsag_manually_unscales_fp16_gradients_and_keeps_amp_scale() -> None:
    source = getsource(RSAGQWDUpdateEngine.step)

    assert "scaler.unscale_" not in source
    assert "_unscale_and_detect_overflow(" in source
    assert "_advance_amp_scaler(scaler, overflow=overflow)" in source


def test_worker_uses_the_same_explicit_amp_scale_for_every_route() -> None:
    source = getsource(_run)

    assert "torch.amp.GradScaler(" in source
    assert "init_scale=args.amp_initial_scale" in source


def test_amp_overflow_skips_optimizer_scheduler_and_model_publication() -> (
    None
):
    native = getsource(NativeUpdateEngine.step)
    rsag = getsource(RSAGQWDUpdateEngine.step)
    worker = getsource(_run)

    assert '"skipped": skipped' in native
    assert '"skipped": True' in rsag
    assert "_advance_amp_scaler(scaler, overflow=overflow)" in rsag
    assert 'if not bool(update["skipped"]):' in worker
    assert "scheduler.step()" in worker


def test_worker_enforces_all_resume_facts_against_uninterrupted_oracle() -> (
    None
):
    source = getsource(_run)

    assert "_write_resume_oracle(" in source
    assert "_load_resume_oracle(" in source
    assert "next_batch_indices=batch_indices" in source
    assert "learning_rate=resume_learning_rate" in source
    assert "amp_scale=resume_amp_scale" in source
    assert "optimizer_state_sha256=resume_optimizer_sha256" in source
    assert "model_sha256=resume_model_sha256" in source
    assert "next_loss=loss_value" in source
    assert "post_optimizer_state_sha256=optimizer_sha256" in source
    assert "post_model_sha256=model_sha256" in source
    assert "assert_resume_matches(resume_oracle, resumed_facts)" in source


def test_checkpoint_restores_rng_byte_tensors_on_the_required_cpu_device() -> (
    None
):
    source = getsource(_load_checkpoint)

    assert 'torch.set_rng_state(payload["rng"]["torch"].cpu())' in source
    assert 'state.cpu() for state in payload["rng"]["cuda"]' in source


def test_rsag_checkpoint_round_trips_live_optimizer_learning_rates() -> None:
    save_source = getsource(RSAGQWDUpdateEngine.state_dict)
    load_source = getsource(RSAGQWDUpdateEngine.load_state_dict)

    assert '"learning_rates": tuple(' in save_source
    assert (
        'float(group["lr"]) for group in self.optimizer.param_groups'
        in save_source
    )
    assert 'group["lr"] = learning_rate' in load_source


def test_rsag_reuses_task2_sharded_adamw_for_master_moments_and_step() -> None:
    init_source = getsource(RSAGQWDUpdateEngine.__init__)
    step_source = getsource(RSAGQWDUpdateEngine._adamw_candidate)

    assert "self.sharded_optimizer = ShardedAdamW(" in init_source
    assert "candidate.step_prevalidated(reduced_shard)" in step_source
    assert "weight_decay=0.0" in init_source
    assert "self.step_count += 1" not in getsource(RSAGQWDUpdateEngine.step)


def test_task2_sharded_adamw_accepts_same_device_cuda_without_moving_state(
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the ShardedAdamW device contract")
    layout = ShardLayout.build(3, 2, 0)
    master = torch.tensor([1.0, -2.0], device="cuda", dtype=torch.float32)
    gradient = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float32)
    optimizer = ShardedAdamW(
        layout,
        master,
        learning_rate=0.1,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )

    optimizer.step(gradient)

    assert optimizer.master.device == master.device
    assert optimizer.exp_avg.device == master.device
    assert optimizer.exp_avg_sq.device == master.device

    cpu_optimizer = ShardedAdamW(
        layout,
        master.cpu(),
        learning_rate=0.1,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    with pytest.raises(ValueError, match="optimizer state device"):
        cpu_optimizer.step(gradient)


def test_cag_hook_casts_fp32_ddp_buckets_to_the_fp16_plan_contract() -> None:
    source = getsource(_register_ddp_hook)

    assert "compressed = buffer.to(dtype=torch.float16).contiguous()" in source
    assert "plan.execute(compressed).wait()" in source


def test_ddp_hook_binds_runtime_grad_bucket_and_future_annotations() -> None:
    source = getsource(_register_ddp_hook)

    assert (
        'hook.__annotations__["bucket"] = torch.distributed.GradBucket'
        in source
    )
    assert (
        'hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]'
    ) in source


def test_worker_blocks_legacy_psi_update_modules_at_the_import_seam() -> None:
    parents = ("psi_policy", "psi_policy.communication")
    names = parents + (
        "psi_policy.communication.ccdl_ddp",
        "psi_policy.communication.ccdl_sharded_adamw",
    )
    previous = {name: sys.modules.get(name) for name in names}
    try:
        policy = ModuleType("psi_policy")
        policy.__path__ = []
        communication = ModuleType("psi_policy.communication")
        communication.__path__ = []
        policy.communication = communication
        sys.modules[parents[0]] = policy
        sys.modules[parents[1]] = communication
        _install_psi_update_seams()

        with pytest.raises(RuntimeError, match="Task 5 worker owns"):
            sys.modules[names[2]].register_ccdl_ddp_hook()
        with pytest.raises(RuntimeError, match="Task 5 worker owns"):
            sys.modules[names[3]].prepare_psi_sharded_adamw()
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_source_manifest_is_stable_read_only_and_excludes_caches(
    tmp_path,
) -> None:
    (tmp_path / "psi_policy").mkdir()
    source = tmp_path / "psi_policy" / "train.py"
    source.write_bytes(b"value = 1\n")
    cache = tmp_path / "psi_policy" / "__pycache__"
    cache.mkdir()
    (cache / "train.pyc").write_bytes(b"cache")
    before = {
        path.relative_to(tmp_path).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in tmp_path.rglob("*")
    }

    first = source_tree_manifest(tmp_path)
    second = source_tree_manifest(tmp_path)

    after = {
        path.relative_to(tmp_path).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in tmp_path.rglob("*")
    }
    assert first == second
    assert first["file_count"] == 1
    assert first["files"] == (
        (
            "psi_policy/train.py",
            "585c93666fcb046b7b264d3fa73202aa2a38254ae82a4b3ba19e873c2d5a9886",
        ),
    )
    assert before == after


def test_step_summary_excludes_warmup_from_steady_performance() -> None:
    records = []
    for step, measured in enumerate((100.0, 2.0, 4.0)):
        record = _step_record()
        record["step"] = step
        record["timing"] = {
            "forward_s": measured / 4.0,
            "backward_s": measured / 4.0,
            "update_s": measured / 4.0,
            "communication_s": measured / 4.0,
            "validation_s": 1000.0,
            "report_serialization_s": 2000.0,
            "measured_s": measured,
        }
        record["communication"]["bytes"] = 10
        records.append(validate_step_record(record))

    summary = summarize_step_records(
        tuple(records),
        warmup_steps=1,
        batch_size_per_rank=16,
        world_size=4,
    )

    assert summary == {
        "steady_samples_per_second": 128.0 / 6.0,
        "step_latency_p50_ms": 3000.0,
        "step_latency_p95_ms": 3900.0,
        "communication_time_s": 26.5,
        "qwd_time_s": 0.0,
        "refresh_time_s": 0.0,
        "communication_bytes": 30,
        "loss_trajectory": (0.5, 0.5, 0.5),
        "rank_gaps": (0.0, 0.0, 0.0),
        "decision_counts": {"native": 3},
    }


# Controller-review regressions. Keep one focused RED per reported finding so
# review closure remains auditable rather than being hidden in one broad test.


def test_review_i1_hashes_the_actual_common_precision_post_engine_model() -> (
    None
):
    source = getsource(_run)

    assert source.index("_convert_model_to_common_fp16(model)") < source.index(
        "engine = build_engine("
    )
    assert source.index("engine = build_engine(") < source.index(
        "initial_sha256 = _parameter_sha256(unwrapped)"
    )


def test_review_i2_rsag_qwd_model_copy_has_exact_world_padded_zero_tail() -> (
    None
):
    source = getsource(RSAGQWDUpdateEngine)

    assert (
        "self.padded_model_numel = self.layout.padded_numel * world_size"
        in source
    )
    assert "self.model_copy_flat[self.global_numel :].zero_()" in source
    assert "self.qwd_plan.execute(" in source
    assert "candidate.master," in source
    assert "self.model_copy_flat," in source


def test_review_i3_cag_plan_identity_includes_stable_bucket_index_and_layout(
) -> None:
    source = getsource(_register_ddp_hook)
    run_source = getsource(_run)

    assert "bucket.index()" in source
    assert "bucket.parameters()" in source
    assert "parameter_layout_by_identity" in source
    assert "bucket_key = (" in source
    assert "plans.get(bucket_key)" in source
    assert "plans.get(buffer.numel())" not in source
    assert "static_graph=True" in run_source


def test_cag_resume_keeps_one_stable_ddp_bucket_across_restart() -> None:
    class Parameter:
        requires_grad = True

        def numel(self) -> int:
            return 1024 * 1024

        def element_size(self) -> int:
            return 2

    model = SimpleNamespace(parameters=lambda: (Parameter(), Parameter()))
    run_source = getsource(_run)

    assert "ddp_bucket_cap_mb = _stable_ddp_bucket_cap_mb(model)" in run_source
    assert "bucket_cap_mb=ddp_bucket_cap_mb" in run_source
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert _stable_ddp_bucket_cap_mb(model) * 1024 * 1024 > trainable_bytes


def test_cag_resume_warms_final_ddp_bucket_order_before_engine_state() -> None:
    run_source = getsource(_run)
    worker_source = Path(
        "tests/benchmarks/distributed_psi_v040_worker.py"
    ).read_text(encoding="utf-8")

    assert run_source.index(
        "telemetry = _register_ddp_hook"
    ) < run_source.index("_stabilize_ddp_bucket_layout(")
    assert run_source.index(
        "_stabilize_ddp_bucket_layout("
    ) < run_source.index("engine = build_engine(")
    assert "_DDP_BUCKET_WARMUP_BACKWARDS = 2" in worker_source
    assert "telemetry.reset_after_warmup()" in worker_source


def test_review_i4_checkpoints_exact_ef_and_resume_compares_post_update() -> (
    None
):
    engine_source = getsource(RSAGQWDUpdateEngine.state_dict)
    load_source = getsource(RSAGQWDUpdateEngine.load_state_dict)
    run_source = getsource(_run)

    assert '"gradient_feedback"' in engine_source
    assert "_restore_plan_feedback(" in load_source
    assert "post_optimizer_state_sha256=" in run_source
    assert "post_model_sha256=" in run_source
    assert run_source.index(
        "update, engine_total_s = _cuda_timed("
    ) < run_source.index("assert_resume_matches(resume_oracle, resumed_facts)")


def test_review_i5_overflow_rolls_back_cag_ef_and_reports_prior_communication(
) -> None:
    engine_source = getsource(NativeUpdateEngine.step)
    worker_source = getsource(_run)

    assert "commit_feedback=not skipped" in engine_source
    assert '"overflow_after_communication"' in engine_source
    assert '"overflow_before_gradient_communication"' in getsource(
        RSAGQWDUpdateEngine.step
    )
    hook_source = getsource(_register_ddp_hook)
    assert "reduce_bucket_overflow" in hook_source
    assert hook_source.index(
        "if bool(bucket_found_inf.item()):"
    ) < hook_source.index("telemetry.snapshot_feedback(bucket_key)")
    assert "failure_facts.append(" in worker_source
    assert (
        parse_args(
            ["--route", "cag", "--inject-overflow-step", "2"]
        ).inject_overflow_step
        == 2
    )


def test_review_i6_resume_oracle_is_optional_unless_verification_requires_it(
) -> None:
    default_args = parse_args(["--route", "native"])
    required_args = parse_args(
        ["--route", "native", "--resume-oracle-mode", "require"]
    )

    assert default_args.resume_oracle_mode == "off"
    assert required_args.resume_oracle_mode == "require"
    source = getsource(_run)
    assert 'if args.resume_oracle_mode == "require":' in source


def test_review_i7_all_routes_share_one_amp_growth_and_backoff_transition(
) -> None:
    source = Path("tests/benchmarks/distributed_psi_v040_worker.py").read_text(
        encoding="utf-8"
    )

    assert source.count("_advance_amp_scaler(scaler, overflow=") == 2
    assert "scaler.update()" not in source
    assert "scaler.update(new_scale=float(scaler.get_scale()))" not in source


def test_review_i8_cuda_timing_and_deferred_raw_rows_are_truthful() -> None:
    source = getsource(_run)

    assert "_cuda_timed(" in source
    assert "torch.cuda.Event(enable_timing=True)" in Path(
        "tests/benchmarks/distributed_psi_v040_worker.py"
    ).read_text(encoding="utf-8")
    assert source.index("validation_loss = _validate_epoch") < source.index(
        "_write_raw_records("
    )


def test_review_i9_nested_schema_and_cross_field_invariants_are_fail_closed(
) -> None:
    telemetry_extra = _task_result()
    telemetry_extra["gpu_telemetry"][0]["extra"] = 1
    with pytest.raises(ValueError, match="gpu_telemetry"):
        validate_task_result(telemetry_extra)

    mismatched = _task_result()
    mismatched["steps"] = len(mismatched["loss_trajectory"]) + 1
    with pytest.raises(ValueError, match="steps"):
        validate_task_result(mismatched)

    telemetry_mismatch = _task_result()
    telemetry_mismatch["gpu_telemetry"][0]["gpu"] = 99
    with pytest.raises(ValueError, match="gpu_telemetry"):
        validate_task_result(telemetry_mismatch)

    source = getsource(_build_workspace)
    assert "_reject_locked_psi_overrides(args.psi_override)" in source
    run_source = getsource(_run)
    assert 'os.environ.get("NVIDIA_VISIBLE_DEVICES"' in run_source


def test_review_m1_validation_is_global_sample_weighted_across_ranks() -> None:
    source = getsource(_validate_epoch)

    assert "loss_sum" in source
    assert "sample_count" in source
    assert "torch.distributed.all_reduce" in source
    assert "median(" not in source


def test_review_m2_training_peak_is_reset_and_captured_before_instrumentation(
) -> None:
    source = getsource(_run)

    assert "torch.cuda.reset_peak_memory_stats(device)" in source
    assert source.index("step_peak_memory_mib =") < source.index(
        "rank_gap = _rank_gap("
    )
    assert "peak_memory_mib=max(engine_peak_memory_mib)" in source


def test_review_m3_rsag_reuses_candidate_buffers_and_prevalidated_step() -> (
    None
):
    init_source = getsource(RSAGQWDUpdateEngine.__init__)
    candidate_source = getsource(RSAGQWDUpdateEngine._adamw_candidate)

    assert "self.candidate_optimizer = ShardedAdamW(" in init_source
    assert "candidate = self.candidate_optimizer" in candidate_source
    assert "candidate.step_prevalidated(reduced_shard)" in candidate_source
    assert "candidate = ShardedAdamW(" not in candidate_source


def test_resume_oracle_round_trips_all_pre_and_post_update_facts(
    tmp_path: Path,
) -> None:
    facts = ResumeFacts(
        next_batch_indices=(3, 7),
        next_batch_sha256="a" * 64,
        next_augmentation_sha256="b" * 64,
        learning_rate=1.0e-4,
        amp_scale=1024.0,
        optimizer_state_sha256="1" * 64,
        model_sha256="2" * 64,
        next_loss=0.5,
        post_learning_rate=9.0e-5,
        post_amp_scale=2048.0,
        post_optimizer_state_sha256="3" * 64,
        post_model_sha256="4" * 64,
    )
    path = tmp_path / "resume.oracle.json"

    _write_resume_oracle(path, facts)

    assert _load_resume_oracle(path) == facts


def test_raw_rows_are_validated_and_accounted_before_publication(
    tmp_path: Path,
) -> None:
    import json

    record = _step_record()
    path = tmp_path / "steps.jsonl"

    _write_raw_records(path, [record])

    published = json.loads(path.read_text(encoding="utf-8"))
    assert published == record
    assert published["timing"]["report_serialization_s"] > 0.0
    validate_step_record(published)


def test_locked_parity_overrides_fail_closed_with_hydra_prefixes() -> None:
    _reject_locked_psi_overrides(
        ["cache=none", "training.torch_compile.enabled=false"]
    )

    for override in ("+training.seed=1", "~training.num_epochs"):
        with pytest.raises(ValueError, match="locked parity"):
            _reject_locked_psi_overrides([override])


class _FakeResidual:
    def __init__(self, value: int):
        self.value = value

    def detach(self) -> _FakeResidual:
        return self

    def clone(self) -> _FakeResidual:
        return _FakeResidual(self.value)


class _FakeFeedbackPlan:
    def __init__(self, residual: _FakeResidual | None):
        self._committed_residual = residual
        self.restored: _FakeResidual | None = None

    def _restore_committed_residual(
        self, residual: _FakeResidual | None
    ) -> None:
        self._committed_residual = residual
        self.restored = residual


def test_cag_overflow_rolls_back_each_bucket_feedback_snapshot() -> None:
    telemetry = _HookTelemetry()
    key = (0, 16, "torch.float16", "cuda", 0)
    plan = _FakeFeedbackPlan(_FakeResidual(7))
    telemetry.plans[key] = plan
    telemetry.snapshot_feedback(key)
    plan._committed_residual = _FakeResidual(9)

    telemetry.consume(commit_feedback=False)

    assert plan.restored is not None
    assert plan.restored.value == 7


class _FakeScaler:
    def __init__(self) -> None:
        self.state = {
            "scale": 8.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2,
            "_growth_tracker": 1,
        }

    def state_dict(self) -> dict[str, object]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.state = dict(state)


def test_shared_amp_transition_has_identical_growth_and_backoff_math() -> None:
    scaler = _FakeScaler()
    _advance_amp_scaler(scaler, overflow=False)
    assert scaler.state["scale"] == 16.0
    assert scaler.state["_growth_tracker"] == 0

    _advance_amp_scaler(scaler, overflow=True)
    assert scaler.state["scale"] == 8.0
    assert scaler.state["_growth_tracker"] == 0


# Second controller re-review: one deterministic RED per finding.


def test_second_review_pending_cag_feedback_binds_before_overflow_return() -> (
    None
):
    source = getsource(_register_ddp_hook)

    assert source.index(
        "telemetry.bind_plan(bucket_key, plan)"
    ) < source.index("if bool(bucket_found_inf.item()):")


def test_second_review_cag_feedback_is_committed_in_unscaled_units() -> None:
    hook_source = getsource(_register_ddp_hook)
    native_source = getsource(NativeUpdateEngine)
    cag_source = getsource(CAGUpdateEngine)

    assert "amp_scale: _AmpScaleState" in hook_source
    assert "1.0 / amp_scale.value" in hook_source
    assert "scaler.get_scale" not in hook_source
    assert "gradients_are_unscaled" in native_source
    assert "gradients_are_unscaled = True" in cag_source


def test_second_review_ddp_warmup_replays_batch_and_restores_module_flags(
) -> None:
    warmup_source = getsource(
        sys.modules[
            "tests.benchmarks.distributed_psi_v040_worker"
        ]._stabilize_ddp_bucket_layout
    )
    run_source = getsource(_run)

    assert "train_loader" not in warmup_source
    assert "batch: object" in warmup_source
    assert "module_training" in warmup_source
    assert "module.training = training" in warmup_source
    assert "replay_batch" in run_source
    assert "chain((replay_batch,), replay_iterator)" in run_source


def test_second_review_epoch_timer_excludes_quality_and_checkpoint_work() -> (
    None
):
    source = getsource(_run)

    assert "epoch_train_s +=" in source
    assert source.index("epoch_train_s +=") < source.index(
        "quality_start = time.perf_counter()"
    )
    assert "time.perf_counter() - epoch_started" not in source


def test_second_review_locked_overrides_reject_ancestors_and_descendants() -> (
    None
):
    for override in (
        "training={seed:1}",
        "+training={seed:1}",
        "~training",
        "train_dataloader={batch_size:8}",
        "communication={enabled:true}",
        "training.seed.child=1",
    ):
        with pytest.raises(ValueError, match="locked parity"):
            _reject_locked_psi_overrides([override])


def test_second_review_mid_epoch_stop_saves_exact_resume_position() -> None:
    source = getsource(_run)

    assert "stopped_mid_epoch" in source
    assert 'f"step-{global_step}-rank{rank}.pt"' in source
    assert "epoch=epoch" in source
    assert "step_in_epoch=batch_index + 1" in source
    assert "checkpoint_next_batch_indices" in source


def test_second_review_feedback_restore_requires_exact_cuda_index() -> None:
    restore_source = getsource(
        sys.modules[
            "tests.benchmarks.distributed_psi_v040_worker"
        ]._restore_plan_feedback
    )
    rsag_source = getsource(RSAGQWDUpdateEngine)

    assert "expected_device" in restore_source
    assert "residual.device != expected_device" in restore_source
    assert "self.master.device.type" in rsag_source
    assert "self.master.device.index" in rsag_source


def test_second_review_gpu_telemetry_joins_rows_by_reported_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules["tests.benchmarks.distributed_psi_v040_worker"]
    output = "2, 20, 200, 42, 1200\n1, 10, 100, 41, 1100\n"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    facts = module._gpu_telemetry((1, 2))

    assert facts[0] == {
        "gpu": 1,
        "utilization": 10.0,
        "memory_used_mib": 100.0,
        "temperature_c": 41.0,
        "sm_clock_mhz": 1100.0,
    }
    assert facts[1]["gpu"] == 2
    assert facts[1]["utilization"] == 20.0


# Third controller re-review: one deterministic RED per finding.


def test_third_review_amp_facts_include_effective_resume_configuration() -> (
    None
):
    result = _task_result()
    run_source = getsource(_run)
    checkpoint_source = getsource(
        sys.modules[
            "tests.benchmarks.distributed_psi_v040_worker"
        ]._save_checkpoint
    )

    assert set(result["amp_configuration"]) == {
        "precision",
        "enabled",
        "initial_scale",
        "effective_start_scale",
        "growth_interval",
        "growth_factor",
        "backoff_factor",
    }
    assert '"amp_configuration"' in checkpoint_source
    assert "effective_start_scale = amp_scale.value" in run_source

    for enabled, initial_scale, effective_start_scale in (
        (False, 65536.0, 65536.0),
        (True, 0.0, 65536.0),
        (True, 65536.0, 0.0),
    ):
        configuration = (
            "fp16",
            enabled,
            initial_scale,
            effective_start_scale,
            2000,
            2.0,
            0.5,
        )
        with pytest.raises(ValueError, match="amp_configuration"):
            replace(_parity_facts(), amp_configuration=configuration)

    checkpoint_amp = dict(result["amp_configuration"])
    checkpoint_amp["initial_scale"] = float("inf")
    scaler_state = {
        "scale": 65536.0,
        "growth_interval": 2000,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "_growth_tracker": 0,
    }
    with pytest.raises(ValueError, match="checkpoint AMP"):
        _validate_amp_configuration(checkpoint_amp, scaler_state=scaler_state)


def test_third_review_checkpoint_preserves_exact_loader_rng_continuation() -> (
    None
):
    run_source = getsource(_run)
    workspace_source = getsource(_build_workspace)
    validation_source = getsource(_validate_epoch)
    augmentation_hash_source = getsource(
        sys.modules[
            "tests.benchmarks.distributed_psi_v040_worker"
        ]._reporting_augmentation_sha256
    )

    assert "train_dataloader.num_workers=0" in workspace_source
    assert "train_dataloader.persistent_workers=false" in workspace_source
    assert "_resume_loader" in run_source
    assert "batch_index < resume_step_in_epoch" not in run_source
    assert "_resume_loader(train_loader, replay_epoch_indices)" in run_source
    assert "epoch_batches = iter(train_loader)" not in run_source
    assert run_source.index("_save_checkpoint(") < run_source.index(
        "quality_start = time.perf_counter()"
    )
    assert "module_training" in validation_source
    assert "rng" in validation_source
    assert "next_batch_sha256" in ResumeFacts.__slots__
    assert "next_augmentation_sha256" in ResumeFacts.__slots__
    assert "batch_sha256 = _state_sha256(batch)" in run_source
    assert 'forward_values["batch"]' not in run_source
    assert "augmentation_rng" in run_source
    assert "_restore_rng_state(augmentation_rng)" in augmentation_hash_source


def test_third_review_cag_hook_uses_cached_python_amp_scale() -> None:
    hook_source = getsource(_register_ddp_hook)
    run_source = getsource(_run)

    assert "get_scale" not in hook_source
    assert "amp_scale.value" in hook_source
    assert "_AmpScaleState" in run_source


def test_qualification_validation_is_seeded_before_loader_iteration() -> None:
    validation_source = getsource(_validate_epoch)
    run_source = getsource(_run)

    assert "seed: int" in validation_source
    validation_loop = validation_source.index("for batch in loader")
    assert validation_source.index("random.seed(seed)") < validation_loop
    assert validation_source.index("numpy.random.seed(seed)") < (
        validation_source.index("for batch in loader")
    )
    assert validation_source.index("torch.manual_seed(seed)") < (
        validation_source.index("for batch in loader")
    )
    assert "generator.manual_seed(seed)" in validation_source
    assert "model, val_loader, device, args.seed" in run_source


def test_qualification_validation_binds_sampler_to_training_epoch() -> None:
    run_source = getsource(_run)

    validation = run_source.index("validation_loss = _validate_epoch")
    sampler_epoch = run_source.rindex(
        "val_sampler.set_epoch(epoch)", 0, validation
    )
    assert sampler_epoch < validation
    assert "del val_sampler" not in run_source


def test_fourth_review_releases_reporting_replay_before_next_peak_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    worker = sys.modules["tests.benchmarks.distributed_psi_v040_worker"]
    replay_hash = getattr(worker, "_reporting_augmentation_sha256", None)
    assert callable(replay_hash), (
        "reporting replay needs an isolated lifetime helper"
    )

    helper_source = getsource(replay_hash)
    run_source = getsource(_run)
    assert "finally:" in helper_source
    assert helper_source.index("del augmented_batch") < helper_source.index(
        "_restore_rng_state(current_rng)"
    )
    assert "augmented_batch" not in run_source
    assert run_source.index("torch.cuda.reset_peak_memory_stats(device)") < (
        run_source.index("step_peak_memory_mib =")
    )
    assert run_source.index("step_peak_memory_mib =") < run_source.index(
        "augmentation_sha256 = _reporting_augmentation_sha256("
    )

    class AugmentedBatch:
        pass

    class Workspace:
        augmented_ref: object | None = None

        def _apply_train_augmentation(self, batch: object) -> object:
            augmented = AugmentedBatch()
            self.augmented_ref = weakref.ref(augmented)
            return augmented

    workspace = Workspace()
    augmentation_rng = object()
    current_rng = object()
    release_observations: list[bool] = []

    def observe_restore(state: object) -> None:
        if state is current_rng:
            assert workspace.augmented_ref is not None
            release_observations.append(workspace.augmented_ref() is None)

    monkeypatch.setattr(worker, "_to_device", lambda batch, device: batch)
    monkeypatch.setattr(worker, "_state_sha256", lambda value: "a" * 64)
    monkeypatch.setattr(worker, "_restore_rng_state", observe_restore)
    digest = replay_hash(
        workspace=workspace,
        batch=object(),
        device=object(),
        augmentation_rng=augmentation_rng,
        current_rng=current_rng,
    )

    assert len(digest) == 64
    assert release_observations == [True]
