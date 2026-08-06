from __future__ import annotations

import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    dequantize_tensor,
    dequantize_reduce_tensors,
    inplace_dequantize_gathered,
    inplace_dequantize_reduce_mean_requantize,
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


@pytest.mark.parametrize("dtype_name", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("num_inputs", (1, 2, 4, 8))
def test_fused_requantize_matches_established_operator_chain(
    extension_status,
    dtype_name: str,
    num_inputs: int,
) -> None:
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype_name]
    config = CompressionConfig(bit=8, group_size=64, topk=0, compact=False)
    sources = [
        torch.linspace(-3.0 + rank * 0.125, 3.0 + rank * 0.125, 128, device="cuda", dtype=dtype)
        for rank in range(num_inputs)
    ]
    payloads = [quantize_tensor(source, config, extension_status=extension_status) for source in sources]
    reduced = dequantize_reduce_tensors(
        payloads,
        (128,),
        config,
        dtype=dtype_name,
        extension_status=extension_status,
        reduce="mean",
    )
    reference = quantize_tensor(reduced, config, extension_status=extension_status)
    payload_stride = ((reference.numel() + 15) // 16) * 16
    output = torch.full((payload_stride,), 0xA5, device="cuda", dtype=torch.uint8)

    assert inplace_dequantize_reduce_mean_requantize(
        payloads,
        output,
        config,
        dtype=dtype_name,
        extension_status=extension_status,
        divisor=num_inputs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output[: reference.numel()], reference, rtol=0, atol=0)
    assert torch.count_nonzero(output[reference.numel() :]).item() == 0


@pytest.mark.parametrize(
    "config",
    (
        CompressionConfig(bit=4, allow_experimental=True),
        CompressionConfig(group_size=32),
        CompressionConfig(topk=1),
        CompressionConfig(compact=True),
    ),
)
def test_fused_requantize_declines_non_fast_path_policies(extension_status, config) -> None:
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(source, config, extension_status=extension_status)
    output = torch.empty(80, device="cuda", dtype=torch.uint8)

    assert not inplace_dequantize_reduce_mean_requantize(
        [payload],
        output,
        config,
        dtype="fp16",
        extension_status=extension_status,
        divisor=1,
    )


def test_fused_requantize_rejects_more_than_eight_inputs(extension_status) -> None:
    config = CompressionConfig()
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(source, config, extension_status=extension_status)
    output = torch.empty(80, device="cuda", dtype=torch.uint8)

    assert not inplace_dequantize_reduce_mean_requantize(
        [payload] * 9,
        output,
        config,
        dtype="fp16",
        extension_status=extension_status,
        divisor=9,
    )


@pytest.mark.parametrize("dtype_name", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_one_launch_dequantizes_every_rank_strided_payload(
    extension_status,
    dtype_name: str,
    world_size: int,
) -> None:
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype_name]
    config = CompressionConfig()
    sources = [torch.randn(128, device="cuda", dtype=dtype) for _ in range(world_size)]
    payloads = [quantize_tensor(source, config, extension_status=extension_status) for source in sources]
    payload_numel = payloads[0].numel()
    payload_stride = ((payload_numel + 15) // 16) * 16
    gathered = torch.zeros(world_size * payload_stride, device="cuda", dtype=torch.uint8)
    for rank, payload in enumerate(payloads):
        gathered[rank * payload_stride : rank * payload_stride + payload_numel].copy_(payload)
    output = torch.empty(world_size * 128, device="cuda", dtype=dtype)
    reference = torch.cat(
        [
            dequantize_tensor(
                payload,
                (128,),
                config,
                dtype=dtype_name,
                extension_status=extension_status,
            )
            for payload in payloads
        ]
    )

    assert inplace_dequantize_gathered(
        gathered,
        output,
        config,
        dtype=dtype_name,
        extension_status=extension_status,
        world_size=world_size,
        payload_numel=payload_numel,
        payload_stride=payload_stride,
        shard_numel=128,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_gathered_dequantize_declines_invalid_native_layouts(extension_status) -> None:
    module = extension_status.module
    native = module.inplace_dequantize_gathered
    config_args = (64, 0, 8, module.QuantType.Linear, False, module.DType.FP16)
    gathered = torch.empty(160, device="cuda", dtype=torch.uint8)
    output = torch.empty(128, device="cuda", dtype=torch.float16)

    assert not native(gathered, output, *config_args, 0, 66, 80, 64)
    assert not native(gathered, output, *config_args, 9, 66, 80, 64)
    assert not native(gathered, output, *config_args, 2, 66, 64, 64)
    assert not native(gathered[:-1], output, *config_args, 2, 66, 80, 64)
    assert not native(gathered, output.to(torch.float32), *config_args, 2, 66, 80, 64)
    assert not native(gathered, output[:-1], *config_args, 2, 66, 80, 64)
    assert not native(gathered, output, 64, 0, 8, module.QuantType.Linear, True, module.DType.FP16, 2, 66, 80, 64)
