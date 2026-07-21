import json
from datetime import datetime, timezone
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields):
        record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def mark_completed(output_dir: str | Path, fields: dict):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    record = {"status": "success", "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    (path / "COMPLETED.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
