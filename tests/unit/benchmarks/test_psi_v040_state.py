"""CPU contracts for the v0.4.0 PSI benchmark training state."""

from __future__ import annotations

import pytest

from tests.benchmarks.psi_v040_state import (
    CommittedResidual,
    QWDSchedule,
    ShardLayout,
)


class IntSubclass(int):
    """An integer subtype rejected by exact state contracts."""


@pytest.mark.parametrize("global_numel", [0, 1, 67])
@pytest.mark.parametrize("world_size", [2, 4])
def test_shard_layout_owns_every_global_element_once(
    global_numel: int,
    world_size: int,
) -> None:
    layouts = [
        ShardLayout.build(global_numel, world_size, rank)
        for rank in range(world_size)
    ]

    assert {layout.padded_numel for layout in layouts} == {
        (global_numel + world_size - 1) // world_size
    }
    assert [layout.start for layout in layouts] == sorted(
        layout.start for layout in layouts
    )
    owned = [
        index
        for layout in layouts
        for index in range(layout.start, layout.start + layout.valid_numel)
    ]
    assert owned == list(range(global_numel))


def test_shard_layout_assigns_uneven_tail_to_last_rank() -> None:
    layouts = [ShardLayout.build(67, 4, rank) for rank in range(4)]

    assert [(layout.start, layout.valid_numel) for layout in layouts] == [
        (0, 17),
        (17, 17),
        (34, 17),
        (51, 16),
    ]
    assert {layout.padded_numel for layout in layouts} == {17}


def test_shard_layout_direct_construction_rejects_forged_ownership() -> None:
    with pytest.raises(ValueError, match="padded_numel"):
        ShardLayout(
            global_numel=67,
            world_size=4,
            rank=0,
            start=0,
            valid_numel=17,
            padded_numel=16,
        )


@pytest.mark.parametrize(
    ("global_numel", "world_size", "rank"),
    [
        (-1, 2, 0),
        (True, 2, 0),
        (IntSubclass(1), 2, 0),
        (1, 0, 0),
        (1, True, 0),
        (1, IntSubclass(2), 0),
        (1, 2, -1),
        (1, 2, 2),
        (1, 2, True),
        (1, 2, IntSubclass(0)),
    ],
)
def test_shard_layout_rejects_invalid_exact_integer_inputs(
    global_numel: object,
    world_size: object,
    rank: object,
) -> None:
    with pytest.raises(ValueError):
        ShardLayout.build(global_numel, world_size, rank)  # type: ignore[arg-type]


def test_shard_layout_accepts_signed_64_bit_max_without_intermediate_overflow(
) -> None:
    layouts = [ShardLayout.build((1 << 63) - 1, 2, rank) for rank in range(2)]

    assert [(layout.start, layout.valid_numel) for layout in layouts] == [
        (0, 1 << 62),
        (1 << 62, (1 << 62) - 1),
    ]
    assert {layout.padded_numel for layout in layouts} == {1 << 62}


@pytest.mark.parametrize(
    ("global_numel", "world_size"),
    [(1 << 63, 2), (1, 1 << 63)],
)
def test_shard_layout_rejects_signed_64_bit_domain_overflow(
    global_numel: int,
    world_size: int,
) -> None:
    with pytest.raises(OverflowError):
        ShardLayout.build(global_numel, world_size, 0)


def test_qwd_schedule_refreshes_at_step_zero_and_each_100_steps() -> None:
    schedule = QWDSchedule(refresh_interval=100)

    assert [schedule.mode(step) for step in (0, 1, 99, 100, 101)] == [
        "fp_refresh",
        "qwd",
        "qwd",
        "fp_refresh",
        "qwd",
    ]


def test_qwd_schedule_force_refresh_overrides_the_regular_cadence() -> None:
    schedule = QWDSchedule(refresh_interval=100)

    assert schedule.mode(37, force_refresh=True) == "fp_refresh"


@pytest.mark.parametrize(
    ("refresh_interval", "step", "force_refresh"),
    [
        (0, 0, False),
        (99, 0, False),
        (True, 0, False),
        (IntSubclass(100), 0, False),
        (100, -1, False),
        (100, True, False),
        (100, IntSubclass(1), False),
        (100, 1, 1),
    ],
)
def test_qwd_schedule_rejects_invalid_exact_inputs(
    refresh_interval: object,
    step: object,
    force_refresh: object,
) -> None:
    with pytest.raises(ValueError):
        QWDSchedule(refresh_interval=refresh_interval).mode(  # type: ignore[arg-type]
            step,
            force_refresh=force_refresh,  # type: ignore[arg-type]
        )


def test_residual_abort_preserves_committed_value_until_commit() -> None:
    residual = CommittedResidual((1.0, 2.0))
    residual.prepare((3.0, 4.0))

    residual.abort()
    assert residual.value == (1.0, 2.0)

    candidate = residual.prepare((3.0, 4.0))
    residual.commit(candidate)
    assert residual.value == (3.0, 4.0)


def test_residual_rejects_candidate_not_prepared_by_its_transaction() -> None:
    left = CommittedResidual((0.0,))
    right = CommittedResidual((0.0,))

    candidate = left.prepare((1.0,))

    with pytest.raises(ValueError, match="candidate"):
        right.commit(candidate)


def _torch_state_module() -> tuple[object, object, object, object]:
    torch = pytest.importorskip("torch")
    from tests.benchmarks.psi_v040_state import (
        ShardedAdamW,
        copy_flat_to_parameters,
        flatten_parameter_copy,
    )

    return torch, ShardedAdamW, flatten_parameter_copy, copy_flat_to_parameters


def _new_sharded_adamw(torch: object, adamw_type: object) -> object:
    layout = ShardLayout.build(5, 2, 1)
    return adamw_type(  # type: ignore[operator]
        layout,
        torch.tensor([1.25, -0.75, 99.0], dtype=torch.float32),
        learning_rate=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )


def _unsafe_forged_layout() -> ShardLayout:
    layout = object.__new__(ShardLayout)
    for name, value in {
        "global_numel": 5,
        "world_size": 2,
        "rank": 1,
        "start": 3,
        "valid_numel": 3,
        "padded_numel": 3,
    }.items():
        object.__setattr__(layout, name, value)
    return layout


def test_sharded_adamw_rejects_forged_layout_before_state_mutation() -> None:
    torch, adamw_type, _, _ = _torch_state_module()
    master = torch.tensor([1.25, -0.75, 99.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="layout"):
        adamw_type(  # type: ignore[operator]
            _unsafe_forged_layout(),
            master,
            learning_rate=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    assert torch.equal(master, torch.tensor([1.25, -0.75, 99.0]))


def test_sharded_adamw_matches_fp32_oracle_and_keeps_padding_zero() -> None:
    torch, adamw_type, _, _ = _torch_state_module()
    optimizer = _new_sharded_adamw(torch, adamw_type)
    parameter = torch.nn.Parameter(torch.tensor([1.25, -0.75]))
    oracle = torch.optim.AdamW(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    for gradient in (
        torch.tensor([0.20, -0.30, 10.0], dtype=torch.float32),
        torch.tensor([-0.50, 0.70, 11.0], dtype=torch.float32),
        torch.tensor([0.10, 0.40, 12.0], dtype=torch.float32),
    ):
        optimizer.step(gradient)
        parameter.grad = gradient[:2].clone()
        oracle.step()

    assert optimizer.step_count == 3
    assert torch.allclose(optimizer.master[:2], parameter.detach(), atol=1e-7)
    oracle_state = oracle.state[parameter]
    assert torch.allclose(
        optimizer.exp_avg[:2],
        oracle_state["exp_avg"],
        atol=1e-7,
    )
    assert torch.allclose(
        optimizer.exp_avg_sq[:2],
        oracle_state["exp_avg_sq"],
        atol=1e-7,
    )
    assert torch.equal(optimizer.master[2:], torch.zeros(1))
    assert torch.equal(optimizer.exp_avg[2:], torch.zeros(1))
    assert torch.equal(optimizer.exp_avg_sq[2:], torch.zeros(1))


def test_flat_parameter_helpers_copy_without_aliasing() -> None:
    torch, _, flatten_parameter_copy, copy_flat_to_parameters = (
        _torch_state_module()
    )
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))

    flat = flatten_parameter_copy((first, second))
    flat.add_(10.0)
    assert torch.equal(first.detach(), torch.tensor([1.0, 2.0]))
    assert torch.equal(second.detach(), torch.tensor([3.0]))
    copy_flat_to_parameters(flat, (first, second))

    assert torch.equal(first.detach(), torch.tensor([11.0, 12.0]))
    assert torch.equal(second.detach(), torch.tensor([13.0]))
    assert torch.equal(flat, torch.tensor([11.0, 12.0, 13.0]))


def test_sharded_adamw_checkpoint_state_isolated_from_callers() -> None:
    torch, adamw_type, _, _ = _torch_state_module()
    source = _new_sharded_adamw(torch, adamw_type)
    source.amp_state = {"scale": 1024.0}
    source.rng_state = {"seed": 7}
    checkpoint = source.state_dict()
    checkpoint["master"][0] = 9.0
    checkpoint["amp_state"]["scale"] = 1.0

    assert source.master[0].item() == 1.25
    assert source.amp_state == {"scale": 1024.0}

    restored = _new_sharded_adamw(torch, adamw_type)
    restored.load_state_dict(checkpoint)
    checkpoint["master"][0] = 11.0
    checkpoint["rng_state"]["seed"] = 9

    assert restored.master[0].item() == 9.0
    assert restored.rng_state == {"seed": 7}


def test_sharded_adamw_checkpoint_round_trip_restores_state_and_refreshes() -> None:
    torch, adamw_type, _, _ = _torch_state_module()
    source = _new_sharded_adamw(torch, adamw_type)
    source.step(torch.tensor([0.20, -0.30, 9.0], dtype=torch.float32))
    source.amp_state = {"scale": 1024.0}
    source.rng_state = {"seed": 7}
    checkpoint = source.state_dict()

    restored = _new_sharded_adamw(torch, adamw_type)
    restored.load_state_dict(checkpoint)

    assert torch.equal(restored.master, source.master)
    assert torch.equal(restored.exp_avg, source.exp_avg)
    assert torch.equal(restored.exp_avg_sq, source.exp_avg_sq)
    assert restored.step_count == source.step_count
    assert restored.amp_state == {"scale": 1024.0}
    assert restored.rng_state == {"seed": 7}
    assert restored.force_refresh is True


def test_sharded_adamw_rejects_forged_checkpoint_fields() -> None:
    torch, adamw_type, _, _ = _torch_state_module()
    source = _new_sharded_adamw(torch, adamw_type)
    checkpoint = source.state_dict()
    invalid_values = {
        "master": torch.zeros(4, dtype=torch.float32),
        "exp_avg": torch.zeros(4, dtype=torch.float32),
        "exp_avg_sq": torch.zeros(4, dtype=torch.float32),
        "step_count": True,
        "learning_rate": float("inf"),
        "betas": (0.9, 1.0),
        "eps": 0.0,
        "weight_decay": float("nan"),
        "amp_state": [],
        "rng_state": [],
    }

    for field, invalid_value in invalid_values.items():
        forged = checkpoint.copy()
        forged[field] = invalid_value
        with pytest.raises(ValueError, match=field):
            _new_sharded_adamw(torch, adamw_type).load_state_dict(forged)


@pytest.mark.parametrize("field", ["master", "exp_avg", "exp_avg_sq"])
def test_sharded_adamw_rejects_checkpoint_nonzero_padding_before_mutation(
    field: str,
) -> None:
    torch, adamw_type, _, _ = _torch_state_module()
    source = _new_sharded_adamw(torch, adamw_type)
    checkpoint = source.state_dict()
    checkpoint[field][-1] = 1.0
    target = _new_sharded_adamw(torch, adamw_type)
    before = target.state_dict()

    with pytest.raises(ValueError, match=field):
        target.load_state_dict(checkpoint)

    after = target.state_dict()
    for state_field in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(after[state_field], before[state_field])
    assert after["step_count"] == before["step_count"]
