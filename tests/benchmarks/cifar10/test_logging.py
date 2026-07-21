import json

from benchmarks.cifar10.logging_utils import JsonlLogger


def test_jsonl_logger_writes_one_valid_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path, rank=0).emit("epoch", epoch=3, val_top1=81.25)
    assert json.loads(path.read_text(encoding="utf-8").strip()) == {
        "kind": "epoch",
        "epoch": 3,
        "val_top1": 81.25,
    }


def test_nonzero_rank_does_not_write(tmp_path):
    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path, rank=1).emit("epoch", epoch=0)
    assert not path.exists()
