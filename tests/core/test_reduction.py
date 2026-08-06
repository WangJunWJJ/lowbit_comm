from importlib import import_module, util


class FakeTensor:
    def __init__(self, values):
        self.values = tuple(values)

    def __truediv__(self, divisor):
        return FakeTensor(value / divisor for value in self.values)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


def test_mean_contract_normalizes_transport_sum_once() -> None:
    assert util.find_spec("ccdl_comm.reduction") is not None
    reduction = import_module("ccdl_comm.reduction")

    contract = reduction.ReductionContract(op="mean", world_size=4, transport_output="sum")

    assert contract.transport_op == "sum"
    assert contract.normalize(FakeTensor([8.0])) == FakeTensor([2.0])


def test_mean_contract_does_not_normalize_transport_mean_again() -> None:
    assert util.find_spec("ccdl_comm.reduction") is not None
    reduction = import_module("ccdl_comm.reduction")

    contract = reduction.ReductionContract(op="mean", world_size=4, transport_output="mean")

    assert contract.transport_op == "mean"
    assert contract.normalize(FakeTensor([2.0])) == FakeTensor([2.0])


def test_sum_contract_rejects_mean_transport_output() -> None:
    assert util.find_spec("ccdl_comm.reduction") is not None
    reduction = import_module("ccdl_comm.reduction")

    try:
        reduction.ReductionContract(op="sum", world_size=4, transport_output="mean")
    except ValueError as exc:
        assert "cannot satisfy sum" in str(exc)
    else:
        raise AssertionError("sum cannot be reconstructed from a transport mean")
