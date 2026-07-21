import pytest

from ccdl_comm.config import CompressionConfig


def test_default_config_targets_safe_8bit_ddp_gradient_compression():
    config = CompressionConfig()

    assert config.bit == 8
    assert config.group_size == 64
    assert config.topk == 0
    assert config.quant_type == "linear"
    assert config.error_feedback is True
    assert config.target == "ddp_gradient_bucket"


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
