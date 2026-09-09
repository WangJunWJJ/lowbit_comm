"""Bind deterministic prefetch to the worker's actual resume protocol."""

from types import SimpleNamespace

import pytest

from tests.benchmarks import psi_training_runtime as runtime


def test_data_configuration_is_bound_in_checkpoint_protocol():
    protocol = runtime.training_protocol(
        rank=0, world_size=2, batch_size=4, native_ddp_mode="standard",
        timing_mode="production", data_mode="deterministic", loader_workers=2,
        loader_prefetch_factor=2, seed=20260821,
    )
    assert protocol["data_pipeline"] == {
        "mode": "deterministic_position_v1", "workers": 2,
        "prefetch_factor": 2, "seed": 20260821,
        "seed_derivation": "sha256-seed-rank-stream-v1",
        "sample_seed_algorithm": "psi-positional-cpu-v1",
    }
    runtime.validate_training_protocol(protocol)
    protocol["data_pipeline"]["workers"] = -1
    with pytest.raises(ValueError, match="protocol"):
        runtime.validate_training_protocol(protocol)


def test_legacy_loader_cannot_silently_enable_workers():
    with pytest.raises(ValueError):
        runtime.training_protocol(
            rank=0, world_size=2, batch_size=4, native_ddp_mode="standard",
            loader_workers=2,
        )


def test_worker_resume_preserves_absolute_position_and_original_dataset():
    torch = pytest.importorskip("torch")
    from tests.benchmarks import distributed_psi_v040_worker as worker

    source = torch.utils.data.DataLoader(torch.arange(12), batch_size=2)
    args = SimpleNamespace(
        data_mode="deterministic", seed=20260821,
        loader_workers=0, loader_prefetch_factor=2,
    )
    template = worker._configure_data_loader(source, args, rank=1, stream=0)
    indices = (8, 1, 1, 5, 0, 9)
    epoch = worker._resume_loader(template, indices, epoch=3)
    assert torch.cat(list(epoch)).tolist() == list(indices)
    resumed = worker._resume_loader(epoch, indices, epoch=3, start_position=2)
    assert torch.cat(list(resumed)).tolist() == list(indices[2:])
    assert resumed._ccdl_source_loader is source


def test_data_cli_requires_explicit_deterministic_mode_for_prefetch():
    from tests.benchmarks.psi_v040_training import parse_args

    args = parse_args(["--route", "native"])
    assert args.data_mode == "legacy" and args.loader_workers == 0
    args = parse_args([
        "--route", "native", "--data-mode", "deterministic", "--loader-workers", "2"
    ])
    assert args.loader_workers == 2


def test_uninstrumented_production_does_not_claim_gradient_timing():
    facts = runtime.measurement_observability("production", "cag", "standard")
    assert facts["gradient_communication_time_available"] is False
    assert facts["scope"] == "controlled_fp16_fp32_master_step_wall"
    assert runtime.measurement_observability(
        "diagnostic", "cag", "standard"
    )["gradient_communication_time_available"] is True


def test_loader_seed_does_not_alias_large_rank_with_next_seed():
    assert runtime.data_loader_seed(7, 32768, 0) != runtime.data_loader_seed(8, 0, 0)
    assert runtime.data_loader_seed(7, 0, 0) != runtime.data_loader_seed(7, 0, 1)
    assert runtime.data_loader_seed(7, 0, 0) == runtime.data_loader_seed(7, 0, 0)


def test_all_new_protocols_bind_common_warmup_and_observation_only_oracle():
    current = runtime.training_protocol(
        rank=0, world_size=2, batch_size=4, native_ddp_mode="standard",
    )
    assert current["version"] == 3
    assert current["model_warmup_backwards"] == 2
    assert current["oracle_policy"] == "observation_only"
    old = {k: v for k, v in current.items() if k not in {
        "timing_mode", "phase_breakdown_available", "core_record_field",
        "model_warmup_backwards", "oracle_policy",
    }}
    old["version"] = 2
    runtime.validate_training_protocol(old)
    assert old != current  # exact checkpoint protocol comparison rejects old resume
