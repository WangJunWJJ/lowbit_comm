from pathlib import Path

import pytest

from examples.training.config import TrainingConfig


def test_default_model_is_the_comparable_4496w_parameter_shape() -> None:
    config = TrainingConfig(mode="native_ddp", synthetic=True)

    assert config.model_parameter_count() == 44_971_744
    assert config.measured_steps == config.steps - config.warmup_steps


def test_config_rejects_ambiguous_or_missing_dataset() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TrainingConfig(mode="native_ddp", synthetic=False)
    with pytest.raises(ValueError, match="exactly one"):
        TrainingConfig(
            mode="native_ddp",
            synthetic=True,
            data_root=Path("dataset"),
        )


@pytest.mark.parametrize("mode", ("native_ddp", "ccdl_sync", "ccdl_async"))
def test_config_accepts_comparable_modes(mode: str) -> None:
    assert TrainingConfig(mode=mode, synthetic=True).mode == mode


def test_config_requires_warmup_before_measured_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be smaller"):
        TrainingConfig(
            mode="native_ddp",
            synthetic=True,
            steps=2,
            warmup_steps=2,
        )
