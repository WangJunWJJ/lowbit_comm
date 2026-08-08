from __future__ import annotations

import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_quantized_buffer,
    dequantize_tensor,
    inplace_dequantize_gathered_add,
    quantize_parameter_delta,
    quantize_tensor,
)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def extension_status():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    status = load_cuda_extension()
    if not status.available:
        pytest.fail(status.reason or "CCDL CUDA extension is unavailable")
    return status


@pytest.mark.parametrize("model_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("shard_numel", (1, 63, 64, 65, 4097))
def test_fused_parameter_delta_quantize_matches_existing_payload(
    extension_status,
    model_dtype,
    shard_numel: int,
) -> None:
    torch.manual_seed(shard_numel)
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    master = torch.randn(shard_numel, device="cuda", dtype=torch.float32)
    model = torch.randn(shard_numel, device="cuda", dtype=model_dtype)
    valid_numel = max(0, shard_numel - 3)
    expected_delta = master - model.float()
    expected_delta[valid_numel:].zero_()
    reference = quantize_tensor(
        expected_delta,
        config,
        extension_status=extension_status,
    )
    output = allocate_quantized_buffer(master, config, dtype="fp32")

    used_fused = quantize_parameter_delta(
        master,
        model,
        config,
        output=output,
        valid_numel=valid_numel,
        extension_status=extension_status,
    )
    torch.cuda.synchronize()

    assert used_fused is True
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_fused_parameter_delta_preserves_tiny_updates(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    model = torch.zeros(64, device="cuda", dtype=torch.float16)
    master = torch.zeros(64, device="cuda", dtype=torch.float32)
    master[:5] = torch.tensor(
        (1.0e-8, -1.0e-7, 5.0e-7, -1.0e-6, 2.0e-6),
        device="cuda",
    )
    output = allocate_quantized_buffer(master, config, dtype="fp32")

    assert quantize_parameter_delta(
        master,
        model,
        config,
        output=output,
        valid_numel=master.numel(),
        extension_status=extension_status,
    )
    restored = dequantize_tensor(
        output,
        master.shape,
        config,
        dtype="fp32",
        extension_status=extension_status,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(restored, master, rtol=0, atol=1.0e-6 / 127.0)


def test_fused_parameter_delta_preserves_non_finite_signal(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    model = torch.zeros(64, device="cuda", dtype=torch.float16)
    master = torch.ones(64, device="cuda", dtype=torch.float32)
    master[11] = float("nan")
    output = allocate_quantized_buffer(master, config, dtype="fp32")

    assert quantize_parameter_delta(
        master,
        model,
        config,
        output=output,
        valid_numel=master.numel(),
        extension_status=extension_status,
    )
    restored = dequantize_tensor(
        output,
        master.shape,
        config,
        dtype="fp32",
        extension_status=extension_status,
    )
    torch.cuda.synchronize()

    assert not torch.isfinite(restored).all()


@pytest.mark.parametrize("model_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("world_size", (1, 2, 4, 8))
def test_fused_gathered_dequantize_add_matches_reference_chain(
    extension_status,
    model_dtype,
    world_size: int,
) -> None:
    torch.manual_seed(world_size)
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    shard_numel = 65
    original_numel = world_size * shard_numel - 3
    model = torch.randn(
        world_size * shard_numel,
        device="cuda",
        dtype=model_dtype,
    )
    payloads = []
    decoded = []
    for rank in range(world_size):
        delta = torch.randn(shard_numel, device="cuda", dtype=torch.float32)
        valid = max(
            0,
            min(shard_numel, original_numel - rank * shard_numel),
        )
        delta[valid:].zero_()
        payload = quantize_tensor(delta, config, extension_status=extension_status)
        payloads.append(payload)
        decoded.append(
            dequantize_tensor(
                payload,
                (shard_numel,),
                config,
                dtype="fp32",
                extension_status=extension_status,
            )
        )
    payload_numel = payloads[0].numel()
    payload_stride = payload_numel
    gathered = torch.zeros(
        world_size * payload_stride,
        device="cuda",
        dtype=torch.uint8,
    )
    for rank, payload in enumerate(payloads):
        gathered[
            rank * payload_stride : rank * payload_stride + payload_numel
        ].copy_(payload)
    expected = model.clone()
    expected[:original_numel].add_(torch.cat(decoded)[:original_numel])

    used_fused = inplace_dequantize_gathered_add(
        gathered,
        model,
        config,
        extension_status=extension_status,
        world_size=world_size,
        payload_numel=payload_numel,
        payload_stride=payload_stride,
        shard_numel=shard_numel,
        original_numel=original_numel,
    )
    torch.cuda.synchronize()

    assert used_fused is True
    torch.testing.assert_close(model, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config",
    (
        CompressionConfig(bit=4, allow_experimental=True, compact=True),
        CompressionConfig(group_size=32, compact=True),
        CompressionConfig(topk=1, compact=True),
        CompressionConfig(bit=8, group_size=64, compact=False),
    ),
)
def test_qwd_kernels_decline_unsupported_policies(
    extension_status,
    config: CompressionConfig,
) -> None:
    master = torch.randn(64, device="cuda", dtype=torch.float32)
    model = torch.randn(64, device="cuda", dtype=torch.float16)
    output = torch.empty(80, device="cuda", dtype=torch.uint8)

    assert not quantize_parameter_delta(
        master,
        model,
        config,
        output=output,
        valid_numel=64,
        extension_status=extension_status,
    )


def test_gathered_add_rejects_invalid_tensor_contract(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    payload = torch.zeros(68, dtype=torch.uint8)
    output = torch.zeros(64, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="input must be a CUDA tensor"):
        inplace_dequantize_gathered_add(
            payload,
            output,
            config,
            extension_status=extension_status,
            world_size=1,
            payload_numel=68,
            payload_stride=68,
            shard_numel=64,
            original_numel=64,
        )
