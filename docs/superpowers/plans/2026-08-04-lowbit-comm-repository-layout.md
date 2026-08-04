# lowbit_comm Repository Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the pre-refactor CCDL tree and make the current refactored
implementation the direct root project of the `lowbit_comm` Git repository.

**Architecture:** Preserve the existing Git history and `wj_dev` branch. Add a
root-layout contract test first, then atomically remove the legacy root tree and
move the tracked contents of `ccdl_comm_refactor/` to the root. Keep the Python
namespace and wheel names unchanged, and validate both pure-Python packaging and
the native CUDA wheel from the flattened root.

**Tech Stack:** Git, Python 3.10+, pytest, Ruff, setuptools/build, PyTorch CUDA
extension, NVIDIA RTX A6000.

## Global Constraints

- The Git repository and README project identity are `lowbit_comm`.
- The Python import namespace remains `ccdl_comm`.
- Distribution names remain `ccdl-core`, `ccdl-cuda`, `ccdl-ascend`, and
  `ccdl-comm`.
- Delete the complete pre-refactor source, tests, benchmarks, and documentation;
  Git history is the only legacy archive.
- Preserve `.git/`, `.gitattributes`, `origin`, `wj_dev`, and all commit history.
- Do not commit caches, egg-info, build directories, `.superpowers`, or generated
  temporary files.
- Do not use `git reset --hard`, `git checkout --`, or rewrite published history.
- The delete and move are one atomic implementation commit.
- Use the existing `CONTRIBUTING.md` commit format.

---

### Task 1: Add the repository-root layout contract

**Files:**
- Create: `ccdl_comm_refactor/tests/packaging/test_repository_layout.py`

**Interfaces:**
- Consumes: the nearest ancestor containing `.git`.
- Produces: `_repository_root(start: Path) -> Path` and layout assertions that
  remain valid after the test file moves to `tests/packaging/`.

- [ ] **Step 1: Write the failing layout test**

```python
from __future__ import annotations

from pathlib import Path


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").is_dir():
            return candidate
    raise RuntimeError("cannot locate lowbit_comm Git root")


def test_refactored_project_is_the_repository_root() -> None:
    root = _repository_root(Path(__file__).resolve())

    assert (root / "ccdl_comm" / "__init__.py").is_file()
    assert (root / "packages" / "ccdl-core" / "pyproject.toml").is_file()
    assert (root / "tests" / "packaging" / "test_core_wheel.py").is_file()
    assert (root / "CONTRIBUTING.md").is_file()
    assert (root / "README.md").read_text(encoding="utf-8").splitlines()[0] == "# lowbit_comm"


def test_legacy_and_staging_trees_are_absent() -> None:
    root = _repository_root(Path(__file__).resolve())

    for relative in (
        "ccdl_comm_refactor",
        "ccdl",
        "csrc",
        "benchmarks",
        "CODE_AUDIT_REPORT.md",
    ):
        assert not (root / relative).exists(), relative
```

- [ ] **Step 2: Run the contract and verify RED**

Run from the current Git root:

```powershell
python -m pytest ccdl_comm_refactor/tests/packaging/test_repository_layout.py -q
```

Expected: both tests fail because the refactored source is still nested and the
legacy/staging trees still exist. The failure must be an assertion failure, not
an import or collection error.

- [ ] **Step 3: Record the pre-migration tracked boundary**

```powershell
git status --short
git ls-files ccdl_comm_refactor | Measure-Object
git ls-files ccdl csrc benchmarks tests docs
```

Expected: only `ccdl_comm_refactor/.superpowers/` is unrelated and untracked;
the refactored tree and legacy tree are both visible as tracked files.

Do not commit the red test separately; it must ship atomically with Task 2.

---

### Task 2: Flatten the refactored project into the Git root

**Files:**
- Delete: `ccdl/`
- Delete: `csrc/`
- Delete: `benchmarks/`
- Delete: legacy `tests/`
- Delete: legacy `docs/`
- Delete: legacy `README.md`
- Delete: legacy `setup.py`
- Delete: legacy `pyproject.toml`
- Delete: `CODE_AUDIT_REPORT.md`
- Move: `ccdl_comm_refactor/ccdl_comm/` to `ccdl_comm/`
- Move: `ccdl_comm_refactor/packages/` to `packages/`
- Move: `ccdl_comm_refactor/tests/` to `tests/`
- Move: `ccdl_comm_refactor/docs/` to `docs/`
- Move: `ccdl_comm_refactor/README.md` to `README.md`
- Move: `ccdl_comm_refactor/CONTRIBUTING.md` to `CONTRIBUTING.md`
- Move: `ccdl_comm_refactor/pyproject.toml` to `pyproject.toml`
- Move: `ccdl_comm_refactor/setup.py` to `setup.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: the approved root-layout design and Task 1 layout contract.
- Produces: a root-installable `lowbit_comm` repository with imports under
  `ccdl_comm` and unchanged distribution names.

- [ ] **Step 1: Verify every destructive target is inside the Git root**

Run in one PowerShell process from the Git root:

```powershell
$repo = (Resolve-Path .).Path
$expected = @(
  "ccdl", "csrc", "benchmarks", "tests", "docs",
  "README.md", "setup.py", "pyproject.toml", "CODE_AUDIT_REPORT.md",
  "ccdl_comm_refactor"
)
foreach ($relative in $expected) {
  $resolved = (Resolve-Path -LiteralPath $relative).Path
  if ($resolved -ne (Join-Path $repo $relative)) {
    throw "unsafe migration target: $resolved"
  }
}
```

Expected: no output and exit code 0. Stop if any resolved path differs.

- [ ] **Step 2: Remove only the tracked legacy tree**

```powershell
git rm -r -- ccdl csrc benchmarks tests docs
git rm -- README.md setup.py pyproject.toml CODE_AUDIT_REPORT.md
```

Expected: Git stages deletion of the old implementation. `.git`,
`.gitattributes`, `.gitignore`, and `ccdl_comm_refactor` remain.

- [ ] **Step 3: Move the tracked refactored project into place**

```powershell
git mv ccdl_comm_refactor/ccdl_comm ccdl_comm
git mv ccdl_comm_refactor/packages packages
git mv ccdl_comm_refactor/tests tests
git mv ccdl_comm_refactor/docs docs
git mv ccdl_comm_refactor/CONTRIBUTING.md CONTRIBUTING.md
git mv ccdl_comm_refactor/README.md README.md
git mv ccdl_comm_refactor/pyproject.toml pyproject.toml
git mv ccdl_comm_refactor/setup.py setup.py
```

Expected: all 526 tracked refactor files are rooted directly under the Git
root. The staging directory contains only ignored or untracked local artifacts.

- [ ] **Step 4: Remove the verified staging-artifact directory**

```powershell
$repo = (Resolve-Path .).Path
$staging = (Resolve-Path -LiteralPath ccdl_comm_refactor).Path
if ($staging -ne (Join-Path $repo "ccdl_comm_refactor")) {
  throw "unsafe staging path: $staging"
}
Remove-Item -LiteralPath $staging -Recurse -Force
```

Expected: `ccdl_comm_refactor/` no longer exists. This removes only untracked
caches, egg-info, build outputs, and `.superpowers` left after `git mv`.

- [ ] **Step 5: Replace stale ignore rules**

Use `apply_patch` so `.gitignore` contains these root-relative rules and no
`ccdl_comm_refactor/` prefix:

```gitignore
__pycache__/
*.py[cod]
*.so
*.egg-info/
build/
dist/
packages/*/build/
.pytest_cache/
.pytest_tmp*/
.ruff_cache/
.superpowers/
benchmark-results/
tests/benchmarks/reports/**/raw/*.nsys-rep
```

- [ ] **Step 6: Update the project identity and active commands**

Use `apply_patch` to make the first README line:

```markdown
# lowbit_comm
```

Search active source, tests, package definitions, README, CONTRIBUTING and
current requirements/design documents:

```powershell
rg -n "ccdl_comm_refactor" ccdl_comm packages tests README.md CONTRIBUTING.md pyproject.toml setup.py
```

Expected: no matches. Remove stale prefixes with focused patches if matches
exist. Historical benchmark report prose may retain an old absolute path only
when it records the environment that actually produced the result.

- [ ] **Step 7: Run the layout contract and verify GREEN**

```powershell
python -m pytest tests/packaging/test_repository_layout.py -q
```

Expected: `2 passed`.

- [ ] **Step 8: Verify the tracked tree has one implementation**

```powershell
git ls-files | Select-String -Pattern "^(ccdl_comm_refactor|ccdl|csrc|benchmarks)/"
git ls-files ccdl_comm packages tests docs README.md CONTRIBUTING.md pyproject.toml setup.py | Measure-Object
git status --short
```

Expected: the first command emits nothing; the second reports the flattened
refactor files; status contains only the intended deletes, renames, layout test,
README and `.gitignore` changes.

- [ ] **Step 9: Run focused validation**

```powershell
python -m pytest tests/packaging -q
python -m ruff check ccdl_comm packages tests/packaging
git diff --check
```

Expected: packaging tests, including the two layout tests, pass; Ruff and diff
checks exit 0.

- [ ] **Step 10: Commit the atomic layout migration**

```powershell
git add -A
git status --short
git commit -m "refactor(repo): flatten lowbit_comm project layout"
```

Before committing, verify no cache, egg-info, build directory, `.superpowers`,
or ignored benchmark artifact is staged.

---

### Task 3: Validate flattened builds locally

**Files:**
- Modify only if a test exposes a root-relative path defect; every fix requires
  a failing regression test first.

**Interfaces:**
- Consumes: flattened root source and multi-wheel build definitions.
- Produces: fresh local evidence that imports, tests, and Core packaging no
  longer depend on the nested path.

- [ ] **Step 1: Run the complete local suite**

```powershell
python -m pytest tests -q
```

Expected: at least the pre-migration local baseline of `746 passed, 29 skipped`,
plus the two new layout tests, with no collection from a legacy `tests/` tree.

- [ ] **Step 2: Build and inspect the Core wheel**

```powershell
python -m build --wheel --no-isolation packages/ccdl-core --outdir dist/core
python -m pytest tests/packaging/test_core_wheel.py tests/packaging/test_install_matrix.py -q
```

Expected: one `ccdl_core-0.1.0-py3-none-any.whl`; it contains `ccdl_comm`
exactly once, contains no native source or extension, and core-only safe import
passes from an isolated directory.

- [ ] **Step 3: Re-run static completion checks**

```powershell
python -m ruff check ccdl_comm packages tests
git diff --check
git status --short
```

Expected: all checks pass. Only fixes prompted by fresh failures may be present.

If Task 3 exposes a defect, stop this plan at the failing command, add a focused
regression test that reproduces that exact defect, and complete its own
red-green cycle before resuming Task 4. Commit that bounded fix as
`fix(build): resolve flattened repository paths`. If no defect is found, do not
create an empty commit.

---

### Task 4: Validate the flattened root on A6000

**Files:**
- Create: `tests/benchmarks/reports/repository_layout/README.md`
- Create: `tests/benchmarks/reports/repository_layout/raw_a6000_layout.json`

**Interfaces:**
- Consumes: committed flattened `wj_dev` source and
  `ccdl-comm-a6000:cu126-torch25`.
- Produces: isolated CUDA build/import/quantization and full-suite evidence.

- [ ] **Step 1: Transfer the exact committed source to a new remote directory**

Create a Git bundle for `wj_dev`, transfer it to the A6000 host, and clone it as:

```text
/home/user/wangjun/lowbit_comm_layout
```

Expected: remote `git rev-parse HEAD` equals local `git rev-parse HEAD`, and the
remote root directly contains `ccdl_comm/`, `packages/`, `tests/`, and `docs/`.

- [ ] **Step 2: Run the A6000 full suite from the repository root**

Inside `ccdl-comm-a6000:cu126-torch25` with the repository mounted at
`/workspace/lowbit_comm`:

```bash
cd /workspace/lowbit_comm
python -m pytest tests -q
```

Expected: no legacy test collection and no regression against the Task 18
A6000 baseline `891 passed, 1 skipped`, plus the two layout tests.

- [ ] **Step 3: Build and install Core plus CUDA wheels in isolation**

```bash
cd /workspace/lowbit_comm
python -m build --wheel --no-isolation packages/ccdl-core --outdir dist/core
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
  python -m build --wheel --no-isolation packages/ccdl-cuda --outdir dist/cuda
rm -rf /tmp/lowbit_comm_layout_install
python -m pip install --no-index --no-deps \
  --target /tmp/lowbit_comm_layout_install \
  dist/core/ccdl_core-*.whl dist/cuda/ccdl_cuda-*.whl
cd /tmp
PYTHONPATH=/tmp/lowbit_comm_layout_install python - <<'PY'
import torch

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor

status = load_cuda_extension()
assert status.available, status.reason
source = torch.linspace(-2, 2, 4096, device="cuda", dtype=torch.float16)
config = CompressionConfig(bit=8, group_size=64, error_feedback=False)
payload = quantize_tensor(source, config, extension_status=status)
restored = dequantize_tensor(
    payload,
    tuple(source.shape),
    config,
    dtype="fp16",
    extension_status=status,
)
error = float((source - restored).abs().max().item())
assert error <= 0.0078125, error
print({"extension_available": True, "max_abs_error": error})
PY
```

Expected: CUDA extension available and INT8 round-trip max absolute error no
worse than the Task 18 value `0.0078125` for the same input.

- [ ] **Step 4: Write the validation evidence**

The JSON must record `status`, the exact 40-character output of
`git rev-parse HEAD`, `repository_root`, `legacy_tree_present`,
`extension_available`, measured `max_abs_error`, measured `pytest_passed`, and
measured `pytest_skipped`. The Markdown report must list
the container, GPU, Torch/CUDA versions, wheel names, commands, test totals,
error value, and every problem found and closed.

- [ ] **Step 5: Commit the validation report**

```powershell
git add tests/benchmarks/reports/repository_layout
git commit -m "test(repo): validate flattened A6000 layout"
```

---

### Task 5: Final verification and push

**Files:**
- No new files expected.

**Interfaces:**
- Consumes: all implementation and validation commits.
- Produces: a clean local branch matching `origin/wj_dev`.

- [ ] **Step 1: Run completion verification**

```powershell
python -m pytest tests/packaging -q
python -m pytest tests -q
python -m ruff check ccdl_comm packages tests
git diff --check
git status --short
```

Expected: all tests and static checks pass; status is clean. Ignored local
artifacts must not appear.

- [ ] **Step 2: Verify repository identity and absence of legacy paths**

```powershell
git remote get-url origin
git branch --show-current
git ls-files | Select-String -Pattern "^(ccdl_comm_refactor|ccdl|csrc|benchmarks)/"
Get-Content README.md -TotalCount 1
```

Expected:

```text
https://github.com/WangJunWJJ/lowbit_comm.git
wj_dev
<no legacy-path output>
# lowbit_comm
```

- [ ] **Step 3: Push and verify the remote branch**

```powershell
git push origin wj_dev
git rev-parse HEAD
git rev-parse origin/wj_dev
```

Expected: both hashes are identical. Do not push directly to the protected main
branch.
