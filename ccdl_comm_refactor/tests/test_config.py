import pytest

from ccdl_comm.config import CompressionConfig


def test_default_config_targets_safe_8bit_tensor_compression():
    config = CompressionConfig()

    assert config.bit == 8
    assert config.group_size == 64
    assert config.topk == 0
    assert config.quant_type == "linear"
    assert config.error_feedback is True
    assert config.target == "tensor"
    assert config.compact is False


@pytest.mark.parametrize("target", ["tensor", "ddp_gradient_bucket", "collective", "p2p"])
def test_config_accepts_independent_communication_targets(target: str) -> None:
    assert CompressionConfig(target=target).target == target


def test_config_rejects_unknown_communication_target() -> None:
    with pytest.raises(ValueError, match="Unsupported target"):
        CompressionConfig(target="optimizer_state")


@pytest.mark.parametrize("bit", [8])
def test_config_accepts_safe_default_bits(bit):
    assert CompressionConfig(bit=bit).bit == bit


@pytest.mark.parametrize("bit", [2, 16])
def test_config_rejects_unsupported_bits(bit):
    with pytest.raises(ValueError, match="bit"):
        CompressionConfig(bit=bit)


@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_config_accepts_supported_group_sizes(group_size):
    assert CompressionConfig(group_size=group_size).group_size == group_size


def test_config_rejects_experimental_4bit_without_explicit_opt_in():
    with pytest.raises(ValueError, match="experimental"):
        CompressionConfig(bit=4)

    assert CompressionConfig(bit=4, allow_experimental=True).bit == 4


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
