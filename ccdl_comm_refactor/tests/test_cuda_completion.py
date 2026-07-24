from ccdl_comm.communication.cuda_completion import CudaCompletionManager, NoopCompletion


def test_noop_completion_is_safe_without_torch() -> None:
    completion = NoopCompletion()

    completion.wait()
    completion.synchronize()


def test_manager_records_event_for_cuda_tensor_with_injected_torch() -> None:
    calls = []

    class FakeEvent:
        def record(self):
            calls.append("record")

        def wait(self):
            calls.append("wait")

        def synchronize(self):
            calls.append("synchronize")

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda

    class FakeTensor:
        is_cuda = True

    manager = CudaCompletionManager(torch_provider=lambda: FakeTorch)
    completion = manager.record_for(FakeTensor())
    completion.wait()
    completion.synchronize()

    assert calls == ["record", "wait", "synchronize"]


def test_manager_uses_noop_completion_for_non_cuda_tensor() -> None:
    class FakeTensor:
        is_cuda = False

    manager = CudaCompletionManager(torch_provider=lambda: None)

    assert isinstance(manager.record_for(FakeTensor()), NoopCompletion)
