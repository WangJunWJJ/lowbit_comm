"""Behavioral checks for the controlled PSI training runtime."""

from copy import deepcopy

import pytest

from tests.benchmarks.psi_training_runtime import FP32MasterWeights, RankBatchSampler


class GlobalSampler:
    def __init__(self):
        self.epoch = 0

    def __len__(self):
        return 8

    def __iter__(self):
        return iter([v + 8 * self.epoch for v in (0, 1, 4, 5, 2, 3, 6, 7)])

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_rank_batches_are_disjoint_complete_and_epoch_aware():
    ranks = [
        RankBatchSampler(GlobalSampler(), rank=r, world_size=2, batch_size=2)
        for r in range(2)
    ]
    assert list(ranks[0]) == [0, 1, 2, 3]
    assert list(ranks[1]) == [4, 5, 6, 7]
    assert len(ranks[0]) == len(ranks[1]) == 4
    for sampler in ranks:
        sampler.set_epoch(1)
    assert list(ranks[0]) == [8, 9, 10, 11]
    assert list(ranks[1]) == [12, 13, 14, 15]


@pytest.mark.parametrize(
    "rank,world,batch", [(2, 2, 2), (-1, 2, 2), (0, 0, 2), (0, 2, 0), (True, 2, 2)]
)
def test_invalid_rank_batch_configuration_rejected(rank, world, batch):
    with pytest.raises(ValueError):
        RankBatchSampler(GlobalSampler(), rank=rank, world_size=world, batch_size=batch)


def test_uneven_global_batches_rejected_instead_of_silently_dropping_data():
    with pytest.raises(ValueError, match="divisible"):
        RankBatchSampler(range(7), rank=0, world_size=2, batch_size=2)


def test_sampler_length_lie_is_rejected_before_any_batch_is_consumed():
    class BadSampler(GlobalSampler):
        def __iter__(self):
            return iter(range(7))

    sampler = RankBatchSampler(BadSampler(), rank=0, world_size=2, batch_size=2)
    with pytest.raises(ValueError, match="length"):
        list(sampler)


def _optimizer_pair(torch):
    model = [
        torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float16)),
        torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float16)),
    ]
    refs = [torch.nn.Parameter(p.detach().float()) for p in model]

    def make(params):
        return torch.optim.AdamW(
            [
                {"params": [params[0]], "weight_decay": 0.03},
                {"params": [params[1]], "weight_decay": 0.0},
            ],
            lr=1e-5,
        )

    optimizer = make(model)
    bridge = FP32MasterWeights(optimizer)
    return model, refs, optimizer, make(refs), bridge


def test_fp32_master_matches_reference_adamw_over_sub_half_ulp_updates():
    torch = pytest.importorskip("torch")
    model, refs, optimizer, reference, bridge = _optimizer_pair(torch)
    for step in range(50):
        for i, (p, ref) in enumerate(zip(model, refs)):
            p.grad = torch.full_like(p, (step + i + 1) / 100)
            ref.grad = p.grad.float()
        bridge.prepare_gradients()
        torch.nn.utils.clip_grad_norm_(bridge.master_parameters, 0.25)
        torch.nn.utils.clip_grad_norm_(refs, 0.25)
        optimizer.step()
        reference.step()
        bridge.publish()
        for p, master, ref in zip(model, bridge.master_parameters, refs):
            torch.testing.assert_close(master, ref, rtol=0, atol=0)
            torch.testing.assert_close(p, ref.half(), rtol=0, atol=0)
            assert optimizer.state[master]["exp_avg"].dtype == torch.float32
            assert optimizer.state[master]["exp_avg_sq"].dtype == torch.float32
        bridge.zero_grad()
        assert all(p.grad is None for p in (*model, *bridge.master_parameters))
    assert model[0][0] != 1  # repeated small updates accumulate in the master


def test_master_checkpoint_is_snapshot_and_resumes_exactly():
    torch = pytest.importorskip("torch")
    model, _, optimizer, _, bridge = _optimizer_pair(torch)
    for p in model:
        p.grad = torch.ones_like(p)
    bridge.prepare_gradients()
    optimizer.step()
    bridge.publish()
    master_state, opt_state = bridge.state_dict(), deepcopy(optimizer.state_dict())
    saved = [v.clone() for v in master_state["parameters"]]
    other, _, other_opt, _, restored = _optimizer_pair(torch)
    restored.load_state_dict(master_state)
    other_opt.load_state_dict(opt_state)
    for params, opt, runtime in [
        (model, optimizer, bridge),
        (other, other_opt, restored),
    ]:
        for p in params:
            p.grad = torch.full_like(p, 0.125)
        runtime.prepare_gradients()
        opt.step()
        runtime.publish()
    for a, b, snapshot, original in zip(
        bridge.master_parameters,
        restored.master_parameters,
        master_state["parameters"],
        saved,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(snapshot, original, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "nan", "protocol", "count"])
def test_invalid_master_checkpoint_does_not_partially_mutate(corruption):
    torch = pytest.importorskip("torch")
    model, _, _, _, bridge = _optimizer_pair(torch)
    before = bridge.state_dict()
    state = deepcopy(before)
    state["parameters"][0].add_(1)
    if corruption == "shape":
        state["parameters"][1] = torch.zeros(2)
    elif corruption == "dtype":
        state["parameters"][1] = state["parameters"][1].half()
    elif corruption == "nan":
        state["parameters"][1].fill_(float("nan"))
    elif corruption == "protocol":
        state["version"] = 0
    else:
        state["parameters"].pop()
    with pytest.raises(ValueError):
        bridge.load_state_dict(state)
    for p, master, previous in zip(
        model, bridge.master_parameters, before["parameters"]
    ):
        torch.testing.assert_close(master, previous, rtol=0, atol=0)
        torch.testing.assert_close(p, previous.half(), rtol=0, atol=0)


def test_master_requires_fresh_optimizer_and_rejects_duplicate_parameters():
    torch = pytest.importorskip("torch")
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    with pytest.raises(ValueError, match="fresh"):
        FP32MasterWeights(optimizer)
    optimizer = torch.optim.AdamW([p])
    optimizer.param_groups[0]["params"].append(p)
    with pytest.raises(ValueError, match="duplicate"):
        FP32MasterWeights(optimizer)


def test_loader_rebuild_keeps_rank_sharding_for_train_validation_and_resume():
    torch = pytest.importorskip("torch")
    from tests.benchmarks.distributed_psi_v040_worker import _resume_loader

    dataset = torch.arange(8)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=GlobalSampler())
    for rank in range(2):
        sampler = RankBatchSampler(
            loader.sampler, rank=rank, world_size=2, batch_size=2
        )
        local = _resume_loader(loader, sampler)
        batches = list(local)
        assert len(local) == 2
        assert torch.cat(batches).tolist() == list(range(rank * 4, rank * 4 + 4))
        resumed = _resume_loader(local, tuple(sampler)[2:])
        assert [v.tolist() for v in resumed] == [batches[1].tolist()]


def test_data_wait_timer_excludes_consumer_work_and_counts_no_terminal_fetch():
    from tests.benchmarks.psi_training_runtime import BatchWaitTimer

    now = [0.0]

    def batches():
        for i in range(3):
            now[0] += 0.25
            yield i

    timer = BatchWaitTimer(clock=lambda: now[0])
    for value in timer.iterate(batches()):
        now[0] += 10 + value  # training/audit work must not be charged to loading
    assert timer.wait_s == 0.75


def test_native_standard_reducer_is_default_and_diagnostic_is_explicit():
    from tests.benchmarks.psi_v040_training import parse_args

    assert parse_args(["--route", "native"]).native_ddp_mode == "standard"
    assert (
        parse_args(
            ["--route", "native", "--native-ddp-mode", "diagnostic"]
        ).native_ddp_mode
        == "diagnostic"
    )


def test_new_checkpoint_protocol_binds_rank_geometry_and_measurement():
    from tests.benchmarks.psi_training_runtime import training_protocol

    native = training_protocol(
        rank=0, world_size=2, batch_size=2, native_ddp_mode="standard"
    )
    assert native["version"] == 2
    assert native["optimizer_state_precision"] == "fp32"
    assert native["sampler"] == "psi_global_batches_rank_sharded"
    assert native != training_protocol(
        rank=1, world_size=2, batch_size=2, native_ddp_mode="standard"
    )
    assert native != training_protocol(
        rank=0, world_size=2, batch_size=2, native_ddp_mode="diagnostic"
    )


def test_device_clip_coefficient_matches_fp32_reference():
    torch = pytest.importorskip("torch")
    from tests.benchmarks.psi_training_runtime import norm_clip_coefficient

    for norm_sq in (0.0, 1.0e-12, 0.25, 4.0, 1.0e10):
        value = torch.tensor(norm_sq, dtype=torch.float32)
        coefficient = norm_clip_coefficient(value, 0.5)
        assert isinstance(coefficient, torch.Tensor)
        assert coefficient.device == value.device
        assert float(coefficient) == pytest.approx(
            min(1.0, 0.5 / (norm_sq**0.5 + 1.0e-6))
        )


def test_result_schema_distinguishes_corrected_protocol_from_legacy():
    from tests.unit.benchmarks.test_psi_v040_training import _task_result
    from tests.benchmarks.psi_v040_training import validate_task_result
    from tests.benchmarks.psi_training_runtime import training_protocol

    result = _task_result()
    assert validate_task_result(result)["schema_version"] == 2
    result["schema_version"] = 3
    result["execution_protocol"] = training_protocol(
        rank=0,
        world_size=result["world_size"],
        batch_size=result["batch_size_per_rank"],
        native_ddp_mode="standard",
    )
    assert validate_task_result(result)["schema_version"] == 3
    assert result["execution_protocol"]["qualification_eligible"] is False
    result["execution_protocol"]["qualification_eligible"] = True
    with pytest.raises(ValueError, match="protocol"):
        validate_task_result(result)
    result["execution_protocol"]["qualification_eligible"] = False
    result["execution_protocol"]["world_size"] = 99
    with pytest.raises(ValueError, match="protocol"):
        validate_task_result(result)
    del result["execution_protocol"]
    with pytest.raises(ValueError):
        validate_task_result(result)


def test_checkpoint_legacy_protocol_is_rejected_before_loading_any_state(monkeypatch):
    from tests.benchmarks import distributed_psi_v040_worker as worker
    from types import SimpleNamespace

    monkeypatch.setattr(worker, "_torch", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="checkpoint"):
        worker._load_checkpoint(
            None,
            route="native",
            model=None,
            engine=None,
            scheduler=None,
            scaler=None,
            amp_scale=None,
            train_loader=None,
            payload={"route": "native"},
        )


def test_checkpoint_rank_protocol_mismatch_is_rejected_before_model_mutation(
    monkeypatch,
):
    from tests.benchmarks import distributed_psi_v040_worker as worker
    from types import SimpleNamespace

    monkeypatch.setattr(worker, "_torch", lambda: SimpleNamespace())
    monkeypatch.setattr(worker, "_validate_checkpoint_payload", lambda *a, **k: None)
    engine = SimpleNamespace(training_protocol={"rank": 0})

    class UntouchedModel:
        def load_state_dict(self, state):
            raise AssertionError("must reject protocol before mutation")

    with pytest.raises(ValueError, match="protocol"):
        worker._load_checkpoint(
            None,
            route="native",
            model=UntouchedModel(),
            engine=engine,
            scheduler=None,
            scaler=None,
            amp_scale=None,
            train_loader=None,
            payload={"training_protocol": {"rank": 1}, "model": {}},
        )


@pytest.mark.parametrize("route", ["native", "cag"])
def test_engine_overflow_preserves_master_moments_and_clears_all_gradients(
    monkeypatch, route
):
    torch = pytest.importorskip("torch")
    from tests.benchmarks import distributed_psi_v040_worker as worker

    overflow = [False]
    monkeypatch.setattr(worker, "_cuda_timed", lambda action: (action(), 0.0))
    monkeypatch.setattr(
        worker,
        "_unscale_and_detect_overflow",
        lambda *args, **kwargs: (overflow[0], (), 0.0, 0),
    )
    monkeypatch.setattr(worker, "_advance_amp_scaler", lambda *a, **k: 1.0)
    model = torch.nn.Linear(2, 1).half()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine_class = (
        worker.NativeUpdateEngine if route == "native" else worker.CAGUpdateEngine
    )
    engine = engine_class(
        model=model,
        optimizer=optimizer,
        grad_clip=1.0,
        telemetry=worker._HookTelemetry(),
        amp_scale=worker._AmpScaleState(1.0),
    )
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    assert engine.step(None)["skipped"] is False
    before = worker._state_sha256(engine._audit_state())
    overflow[0] = True
    for p in model.parameters():
        p.grad = torch.full_like(p, float("inf"))
    assert engine.step(None)["skipped"] is True
    assert engine.step_count == 1
    assert worker._state_sha256(engine._audit_state()) == before
    assert all(
        p.grad is None
        for p in (*model.parameters(), *engine.master_weights.master_parameters)
    )
    checkpoint = deepcopy(engine.state_dict())
    for p in engine.master_weights.master_parameters:
        p.data.add_(1)
    engine.load_state_dict(checkpoint)
    assert worker._state_sha256(engine._audit_state()) == before


def test_diagnostic_native_predivides_before_fp16_sum(monkeypatch):
    torch = pytest.importorskip("torch")
    from tests.benchmarks import distributed_psi_v040_worker as worker
    from types import SimpleNamespace

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda buffer: buffer.mul_(2))
    monkeypatch.setattr(worker, "_cuda_timed", lambda action: (action(), 0.0))

    class Model(torch.nn.Linear):
        def register_comm_hook(self, state, hook):
            self.hook = hook

    model = Model(1, 1)
    worker._register_ddp_hook(model, "native", worker._AmpScaleState(1.0))
    value = torch.tensor([60000.0], dtype=torch.float16)
    result = model.hook(None, SimpleNamespace(buffer=lambda: value)).wait()
    assert float(result[0]) == 60000.0


def test_standard_ddp_keeps_its_own_reducer(monkeypatch):
    from tests.benchmarks import distributed_psi_v040_worker as worker

    calls = []
    sentinel = worker._HookTelemetry()
    monkeypatch.setattr(
        worker, "_register_ddp_hook", lambda *a: calls.append(a) or sentinel
    )
    assert (
        worker._configure_ddp_communication(None, "native", None, "standard")
        is not sentinel
    )
    assert calls == []
    assert (
        worker._configure_ddp_communication(None, "native", None, "diagnostic")
        is sentinel
    )
    assert (
        worker._configure_ddp_communication(None, "cag", None, "standard") is sentinel
    )
    assert len(calls) == 2


def test_execution_counters_distinguish_iterations_updates_and_samples():
    from tests.benchmarks.psi_training_runtime import execution_counters

    records = [
        {"batch_indices": [0, 1], "quality": {"optimizer_step": 10, "finite": True}},
        {"batch_indices": [2, 3], "quality": {"optimizer_step": 10, "finite": False}},
        {"batch_indices": [4, 5], "quality": {"optimizer_step": 11, "finite": True}},
    ]
    assert execution_counters(records) == {
        "iterations_completed": 3,
        "successful_optimizer_updates": 2,
        "skipped_updates": 1,
        "optimizer_step_final": 11,
        "local_samples_consumed": 6,
    }
