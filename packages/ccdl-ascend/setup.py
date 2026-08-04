from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY))

from ccdl_comm.build.distributions import ascend_setup_kwargs  # noqa: E402


setup(**ascend_setup_kwargs(PACKAGE_ROOT, os.environ))
