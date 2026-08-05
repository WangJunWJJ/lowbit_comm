from ccdl_comm.quantization.error_feedback import ErrorFeedbackState


class FakeTensor:
    def __init__(self, values, *, detached=False, cloned=False):
        self.values = tuple(values)
        self.detached = detached
        self.cloned = cloned

    def __add__(self, other):
        return FakeTensor(a + b for a, b in zip(self.values, other.values))

    def __sub__(self, other):
        return FakeTensor(a - b for a, b in zip(self.values, other.values))

    def detach(self):
        return FakeTensor(self.values, detached=True, cloned=self.cloned)

    def clone(self):
        return FakeTensor(self.values, detached=self.detached, cloned=True)

    def __eq__(self, other):
        return (
            isinstance(other, FakeTensor)
            and self.values == other.values
            and self.detached == other.detached
            and self.cloned == other.cloned
        )


class StrictShapeTensor(FakeTensor):
    @property
    def shape(self):
        return (len(self.values),)

    def __add__(self, other):
        if self.shape != other.shape:
            raise RuntimeError("shape mismatch")
        return StrictShapeTensor(a + b for a, b in zip(self.values, other.values))

    def __sub__(self, other):
        if self.shape != other.shape:
            raise RuntimeError("shape mismatch")
        return StrictShapeTensor(a - b for a, b in zip(self.values, other.values))

    def detach(self):
        return StrictShapeTensor(self.values, detached=True, cloned=self.cloned)

    def clone(self):
        return StrictShapeTensor(self.values, detached=self.detached, cloned=True)


def test_compensate_returns_original_tensor_without_residual() -> None:
    state = ErrorFeedbackState()
    tensor = FakeTensor([1.0, 2.0])

    assert state.compensate("bucket-0", tensor) is tensor


def test_update_stores_detached_cloned_residual_and_applies_it_next_time() -> None:
    state = ErrorFeedbackState()
    original = FakeTensor([1.0, 2.0])
    transmitted = FakeTensor([0.75, 1.5])

    state.update("bucket-0", original=original, transmitted=transmitted)

    residual = state.get("bucket-0")
    assert residual == FakeTensor([0.25, 0.5], detached=True, cloned=True)
    assert state.compensate("bucket-0", FakeTensor([10.0, 20.0])) == FakeTensor([10.25, 20.5])


def test_update_local_uses_local_reconstruction_instead_of_global_result() -> None:
    state = ErrorFeedbackState()

    state.update_local(
        "bucket-0",
        prepared=FakeTensor([4.0]),
        local_restored=FakeTensor([3.5]),
    )

    assert state.get("bucket-0") == FakeTensor([0.5], detached=True, cloned=True)


def test_clear_removes_one_or_all_residuals() -> None:
    state = ErrorFeedbackState()
    state.update("a", original=FakeTensor([2.0]), transmitted=FakeTensor([1.0]))
    state.update("b", original=FakeTensor([4.0]), transmitted=FakeTensor([1.0]))

    state.clear("a")

    assert state.get("a") is None
    assert state.get("b") == FakeTensor([3.0], detached=True, cloned=True)

    state.clear()

    assert state.get("b") is None


def test_run_cycle_compensates_then_updates_residual() -> None:
    state = ErrorFeedbackState()
    state.update("bucket-0", original=FakeTensor([4.0]), transmitted=FakeTensor([3.0]))

    compensated = state.compensate("bucket-0", FakeTensor([10.0]))
    state.update("bucket-0", original=compensated, transmitted=FakeTensor([10.25]))

    assert compensated == FakeTensor([11.0])
    assert state.get("bucket-0") == FakeTensor([0.75], detached=True, cloned=True)


def test_compensate_discards_residual_when_bucket_shape_changes() -> None:
    state = ErrorFeedbackState()
    state.update("bucket-0", original=StrictShapeTensor([4.0, 5.0]), transmitted=StrictShapeTensor([3.0, 4.0]))
    tensor = StrictShapeTensor([10.0])

    assert state.compensate("bucket-0", tensor) is tensor
    assert state.get("bucket-0") is None
