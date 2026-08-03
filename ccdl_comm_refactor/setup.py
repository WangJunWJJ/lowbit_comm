from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ccdl_comm.build.setuptools import build_setup_kwargs


if __name__ == "__main__":
    setup(**build_setup_kwargs(env=os.environ))
