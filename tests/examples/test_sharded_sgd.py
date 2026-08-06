from __future__ import annotations

import pytest

from ccdl_comm.shard import ReducedShard
from examples.training.sharded_sgd import (
    TorchShardedSgdConsumer,
    compile_torch_shard_layout,
    exact_mean_reduce_scatter,
)

torch = pytest.importorskip("torch")


def parameters() -> tuple[torch.nn.Parameter, ...]:
    return (
        torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        torch.nn.Parameter(torch.tensor([5.0])),
    )


def test_compile_layout_preserves_parameter_order_and_padding() -> None:
    active_parameters = parameters()

    layout = compile_torch_shard_layout(active_parameters, rank=1, world_size=2)

    assert layout.original_numel == 5
    assert layout.padded_numel == 6
    assert layout.shard_numel == 3
    assert layout.logical_range == (3, 5)
    assert tuple(item.offset for item in layout.parameters) == (0, 4)
    assert tuple(item.shape for item in layout.parameters) == ((2, 2), (1,))
    assert layout.dtype == "fp32"


def test_compile_layout_rejects_mixed_dtype() -> None:
    active_parameters = (
        torch.nn.Parameter(torch.ones(2, dtype=torch.float32)),
        torch.nn.Parameter(torch.ones(2, dtype=torch.float64)),
    )

    with pytest.raises(ValueError, match="same dtype"):
        compile_torch_shard_layout(active_parameters, rank=0, world_size=2)


def test_compile_layout_rejects_mixed_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for a mixed-device parameter tuple")
    active_parameters = (
        torch.nn.Parameter(torch.ones(2)),
        torch.nn.Parameter(torch.ones(2, device="cuda")),
    )

    with pytest.raises(ValueError, match="same device"):
        compile_torch_shard_layout(active_parameters, rank=0, world_size=2)


@pytest.mark.parametrize("learning_rate", [True, float("nan"), float("inf")])
def test_consumer_rejects_non_finite_or_non_numeric_learning_rate(
    learning_rate,
) -> None:
    active_parameters = parameters()
    active_layout = compile_torch_shard_layout(
        active_parameters,
        rank=0,
        world_size=1,
    )

    with pytest.raises((TypeError, ValueError), match="finite positive number"):
        TorchShardedSgdConsumer(
            active_parameters,
            layout=active_layout,
            learning_rate=learning_rate,
            all_gather_into_tensor=lambda output, local: None,
            torch=torch,
        )


def test_flatten_gradients_reuses_buffer_and_zeros_missing_gradient() -> None:
    active_parameters = parameters()
    active_parameters[0].grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    layout = compile_torch_shard_layout(active_parameters, rank=0, world_size=2)
    consumer = TorchShardedSgdConsumer(
        active_parameters,
        layout=layout,
        learning_rate=0.1,
        all_gather_into_tensor=lambda output, local: None,
        torch=torch,
    )

    first = consumer.flatten_gradients()
    first_pointer = first.data_ptr()
    active_parameters[0].grad.fill_(2.0)
    second = consumer.flatten_gradients()

    assert second.data_ptr() == first_pointer
    torch.testing.assert_close(second[:4], torch.full((4,), 2.0))
    assert second[4].item() == 0.0
    assert second[5].item() == 0.0
    assert consumer.reduced_output().numel() == layout.shard_numel


def test_consumer_updates_only_valid_local_parameters_and_writes_back() -> None:
    active_parameters = parameters()
    layout = compile_torch_shard_layout(active_parameters, rank=1, world_size=2)
    gather_calls = []

    def gather(output, local) -> None:
        gather_calls.append((output, local))
        output.copy_(torch.cat((torch.tensor([0.9, 1.8, 2.7]), local)))

    consumer = TorchShardedSgdConsumer(
        active_parameters,
        layout=layout,
        learning_rate=0.1,
        all_gather_into_tensor=gather,
        torch=torch,
    )
    reduced = ReducedShard(
        shard=torch.tensor([1.0, 2.0, 99.0]),
        shard_index=1,
        shard_numel=3,
        original_shape=(5,),
        original_numel=5,
        padded_numel=6,
        world_size=2,
        reduce="mean",
        dtype="fp32",
    )

    result = consumer.consume(reduced)

    torch.testing.assert_close(result, torch.tensor([0.9, 1.8, 2.7, 3.9, 4.8]))
    torch.testing.assert_close(
        torch.cat(tuple(parameter.detach().reshape(-1) for parameter in active_parameters)),
        result,
    )
    assert consumer.local_parameters[-1].item() == 0.0
    assert len(gather_calls) == 1
    assert gather_calls[0][0].is_contiguous()
    assert gather_calls[0][1].is_contiguous()


def test_consumer_workspace_pointers_are_stable_across_steps() -> None:
    active_parameters = parameters()
    layout = compile_torch_shard_layout(active_parameters, rank=0, world_size=1)

    def gather(output, local) -> None:
        output.copy_(local)

    consumer = TorchShardedSgdConsumer(
        active_parameters,
        layout=layout,
        learning_rate=0.01,
        all_gather_into_tensor=gather,
        torch=torch,
    )
    expected_pointers = consumer.buffer_pointers()
    for _ in range(10):
        for parameter in active_parameters:
            parameter.grad = torch.ones_like(parameter)
        flat = consumer.flatten_gradients()
        consumer.reduced_output().copy_(flat)
        consumer.consume(
            ReducedShard(
                shard=consumer.reduced_output(),
                shard_index=0,
                shard_numel=5,
                original_shape=(5,),
                original_numel=5,
                padded_numel=5,
                world_size=1,
                reduce="mean",
                dtype="fp32",
            )
        )
        assert consumer.buffer_pointers() == expected_pointers


def test_mismatched_shard_does_not_mutate_parameters() -> None:
    active_parameters = parameters()
    layout = compile_torch_shard_layout(active_parameters, rank=0, world_size=1)
    consumer = TorchShardedSgdConsumer(
        active_parameters,
        layout=layout,
        learning_rate=0.1,
        all_gather_into_tensor=lambda output, local: output.copy_(local),
        torch=torch,
    )
    before = tuple(parameter.detach().clone() for parameter in active_parameters)
    mismatched = ReducedShard(
        shard=torch.ones(6),
        shard_index=0,
        shard_numel=6,
        original_shape=(6,),
        original_numel=6,
        padded_numel=6,
        world_size=1,
        reduce="mean",
        dtype="fp32",
    )

    with pytest.raises(ValueError, match="does not match layout"):
        consumer.consume(mismatched)

    for parameter, reference in zip(active_parameters, before, strict=True):
        torch.testing.assert_close(parameter, reference)


def test_exact_mean_reduce_scatter_uses_rank_local_mean_shard() -> None:
    active_parameters = parameters()
    layout = compile_torch_shard_layout(active_parameters, rank=1, world_size=2)
    flat_gradients = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
    peer_gradients = torch.tensor([5.0, 6.0, 7.0, 8.0, 0.0, 0.0])
    output = torch.empty(layout.shard_numel)
    calls = []

    def reduce_scatter(target, source) -> None:
        calls.append((target, source))
        reduced = flat_gradients + peer_gradients
        target.copy_(reduced.chunk(2)[layout.shard_index])

    reduced = exact_mean_reduce_scatter(
        flat_gradients,
        out=output,
        layout=layout,
        reduce_scatter_tensor=reduce_scatter,
    )

    assert len(calls) == 1
    assert reduced.shard is output
    assert reduced.logical_range == layout.logical_range
    assert reduced.reduce == "mean"
    assert reduced.transport == "exact_reduce_scatter"
    torch.testing.assert_close(output, torch.tensor([6.0, 0.0, 0.0]))
