import pytest

from examples.training.metrics import (
    CorrectnessMetrics,
    ExecutionMetrics,
    MemoryMetrics,
    TimingMetrics,
    TrainingResult,
)


def minimal_training_result() -> TrainingResult:
    return TrainingResult(
        mode="ccdl_async",
        world_size=2,
        global_batch_size=32,
        parameter_count=44_971_744,
        timing=TimingMetrics(
            measured_steps=10,
            elapsed_seconds=2.0,
            step_latencies_ms=(190.0, 210.0) * 5,
            overlap_efficiency=0.5,
        ),
        memory=MemoryMetrics(peak_allocated_bytes=1024),
        losses=(4.0, 3.0),
        correctness=CorrectnessMetrics(
            rank_parameters_consistent=True,
            max_parameter_difference=0.0,
            finite_loss=True,
        ),
        execution=ExecutionMetrics(
            requested_mode="ccdl_async",
            effective_strategy="all_gather",
            capability="cuda_extension",
            fallback_reason=None,
        ),
    )


def test_result_contains_stable_timing_correctness_and_execution_schema() -> None:
    payload = minimal_training_result().to_dict()

    assert payload["timing"]["throughput_samples_per_second"] == 160.0
    assert payload["timing"]["mean_step_latency_ms"] == 200.0
    assert payload["timing"]["overlap_efficiency"] == 0.5
    assert payload["correctness"]["rank_parameters_consistent"] is True
    assert payload["execution"]["fallback_reason"] is None
    assert payload["loss"]["initial"] == 4.0
    assert payload["loss"]["final"] == 3.0


def test_metrics_reject_non_finite_or_out_of_range_measurements() -> None:
    with pytest.raises(ValueError, match="overlap_efficiency"):
        TimingMetrics(
            measured_steps=1,
            elapsed_seconds=1.0,
            step_latencies_ms=(1.0,),
            overlap_efficiency=1.1,
        )
    with pytest.raises(ValueError, match="losses must be finite"):
        TrainingResult(
            mode="native_ddp",
            world_size=1,
            global_batch_size=1,
            parameter_count=1,
            timing=TimingMetrics(1, 1.0, (1.0,), 0.0),
            memory=MemoryMetrics(0),
            losses=(float("nan"),),
            correctness=CorrectnessMetrics(True, 0.0, False),
            execution=ExecutionMetrics("native_ddp", "native", "torch", None),
        )
