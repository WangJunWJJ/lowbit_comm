import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: Path, rank: int):
        self.path = Path(path)
        self.rank = rank

    def emit(self, kind: str, **fields) -> None:
        if self.rank != 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def cuda_elapsed_ms(callable_):
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)
