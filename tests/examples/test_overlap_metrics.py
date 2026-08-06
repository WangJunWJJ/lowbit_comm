import pytest

from examples.training.overlap import (
    CudaOverlapRecorder,
    InvalidOverlapMeasurement,
    OverlapMeasurement,
    classify_overlap,
    merge_intervals,
    measurement_from_intervals,
    measurement_from_interval_sets,
)


def test_overlap_efficiency_uses_executed_timeline() -> None:
    measurement = OverlapMeasurement(
        communication_ms=4.0,
        compute_ms=6.0,
        overlapped_ms=8.0,
        exposed_communication_ms=2.0,
    )

    assert measurement.intersection_ms == 2.0
    assert measurement.overlap_efficiency() == 0.5


def test_async_label_requires_timeline_intersection() -> None:
    assert (
        classify_overlap(future_returned=True, timeline_intersection_ms=0.0)
        == "not_overlapped"
    )
    assert (
        classify_overlap(future_returned=True, timeline_intersection_ms=1.0)
        == "timeline_overlapped"
    )
    assert (
        classify_overlap(future_returned=False, timeline_intersection_ms=1.0)
        == "synchronous"
    )


def test_measurement_merges_overlapping_bucket_intervals_without_double_counting() -> None:
    assert merge_intervals(((1.0, 3.0), (2.0, 5.0), (7.0, 8.0))) == (
        (1.0, 5.0),
        (7.0, 8.0),
    )

    measurement = measurement_from_intervals(
        compute_interval=(0.0, 6.0),
        communication_intervals=((1.0, 3.0), (2.0, 5.0), (7.0, 8.0)),
    )

    assert measurement.communication_ms == 5.0
    assert measurement.compute_ms == 6.0
    assert measurement.intersection_ms == 4.0
    assert measurement.overlapped_ms == 7.0
    assert measurement.exposed_communication_ms == 1.0


def test_measurement_excludes_hook_gap_from_compute_intervals() -> None:
    measurement = measurement_from_interval_sets(
        compute_intervals=((0.0, 2.0), (5.0, 10.0)),
        communication_intervals=((2.0, 8.0),),
    )

    assert measurement.compute_ms == 7.0
    assert measurement.communication_ms == 6.0
    assert measurement.intersection_ms == 3.0
    assert measurement.exposed_communication_ms == 3.0
    assert measurement.overlap_efficiency() == 0.5


@pytest.mark.parametrize(
    "kwargs",
    (
        {"communication_ms": -1.0, "compute_ms": 1.0, "overlapped_ms": 1.0, "exposed_communication_ms": 0.0},
        {"communication_ms": 1.0, "compute_ms": 1.0, "overlapped_ms": 3.0, "exposed_communication_ms": 0.0},
        {"communication_ms": 1.0, "compute_ms": 1.0, "overlapped_ms": 1.0, "exposed_communication_ms": 2.0},
    ),
)
def test_invalid_timeline_measurements_are_rejected(kwargs) -> None:
    with pytest.raises(InvalidOverlapMeasurement):
        OverlapMeasurement(**kwargs)


def test_measured_hook_preserves_pytorch_ddp_type_annotations() -> None:
    class Cuda:
        @staticmethod
        def is_available():
            return True

    class Torch:
        cuda = Cuda()

    def hook(state, bucket):
        return None

    annotations = {
        "state": object,
        "bucket": type("GradBucket", (), {}),
        "return": type("FutureTensor", (), {}),
    }
    hook.__annotations__ = annotations
    recorder = CudaOverlapRecorder(torch=Torch(), enabled=True)

    wrapped = recorder.wrap_hook(hook)

    assert wrapped.__annotations__ == annotations
