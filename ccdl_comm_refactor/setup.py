from __future__ import annotations

import os

from setuptools import setup

from ccdl_comm.build.setuptools import build_setup_kwargs


if __name__ == "__main__":
    setup(**build_setup_kwargs(env=os.environ))
