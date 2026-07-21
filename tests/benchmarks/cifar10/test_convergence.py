from benchmarks.cifar10.aggregate import find_convergence


def test_convergence_requires_five_consecutive_epochs():
    values = [79, 80, 79, 80, 80, 80, 80, 80]
    rows = [
        {
            "epoch": index,
            "val_top1": value,
            "optimizer_steps": 100 * (index + 1),
            "wall_s": 10 * (index + 1),
        }
        for index, value in enumerate(values)
    ]
    assert find_convergence(rows, threshold=80.0, patience=5) == {
        "epoch": 3,
        "optimizer_steps": 400,
        "wall_s": 40,
    }


def test_convergence_returns_none_when_not_sustained():
    rows = [
        {"epoch": index, "val_top1": value, "optimizer_steps": index, "wall_s": index}
        for index, value in enumerate([80, 80, 80, 80, 79])
    ]
    assert find_convergence(rows, threshold=80.0, patience=5) is None
