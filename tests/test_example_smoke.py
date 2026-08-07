from __future__ import annotations

import subprocess
import sys


def test_compressed_sharded_optimizer_help_does_not_require_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.training.compressed_sharded_optimizer",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "sharded_compressed" in result.stdout
