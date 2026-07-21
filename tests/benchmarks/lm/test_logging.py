import json

from benchmarks.lm.logging_utils import JsonlLogger, mark_completed


def test_jsonl_logger_appends_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(path)
    logger.emit("train", step=1, loss=2.0)
    logger.emit("eval", step=1, perplexity=4.0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["train", "eval"]


def test_completion_marker_contains_success(tmp_path):
    mark_completed(tmp_path, {"variant": "nccl_fp32", "seed": 17})
    record = json.loads((tmp_path / "COMPLETED.json").read_text(encoding="utf-8"))
    assert record["status"] == "success"
    assert record["variant"] == "nccl_fp32"
