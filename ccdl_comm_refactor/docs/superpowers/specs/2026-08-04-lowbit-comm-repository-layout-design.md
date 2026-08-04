# lowbit_comm 仓库根目录重构设计

## 1. 目标

将当前 Git 仓库从“旧版 CCDL 位于根目录、重构版位于
`ccdl_comm_refactor/` 子目录”的混合结构，调整为根目录直接承载完整重构版
`lowbit_comm` 工程。

迁移完成后：

- Git 仓库名称和 README 项目名称为 `lowbit_comm`；
- Python 导入命名空间继续使用 `ccdl_comm`；
- wheel 名称继续使用 `ccdl-core`、`ccdl-cuda`、`ccdl-ascend` 和
  `ccdl-comm`；
- 旧版源码、测试、benchmark 和文档不在工作树中保留，仅通过 Git 历史
  追溯；
- 当前 `wj_dev` 分支、远程地址和已有提交历史保持不变。

## 2. 当前问题

当前 Git 根目录同时包含：

- 旧版：`ccdl/`、`csrc/`、`benchmarks/`、`tests/`、`docs/`、旧
  `README.md`、旧 `setup.py` 和旧 `pyproject.toml`；
- 重构版：`ccdl_comm_refactor/ccdl_comm/`、`packages/`、`tests/`、
  `docs/` 及对应构建文件。

这会导致仓库入口、测试口径、构建配置和文档身份不明确，也容易让用户误用
旧版模块。

## 3. 选定方案

采用原地扁平化方案：删除旧版受控文件，然后使用 Git 可识别的移动操作将
`ccdl_comm_refactor/` 中的重构版受控文件迁移到仓库根目录。

不重新初始化 Git，不重写历史，也不创建长期并存的 legacy 目录。

## 4. 目标目录结构

```text
lowbit_comm/
├── .git/
├── .gitattributes
├── .gitignore
├── ccdl_comm/
├── packages/
│   ├── ccdl-core/
│   ├── ccdl-cuda/
│   └── ccdl-ascend/
├── tests/
├── docs/
├── CONTRIBUTING.md
├── README.md
├── pyproject.toml
└── setup.py
```

根目录不得继续存在：

- `ccdl_comm_refactor/`；
- 旧版 `ccdl/`；
- 旧版根级 `csrc/`；
- 旧版 `benchmarks/`；
- `CODE_AUDIT_REPORT.md`。

原生源码继续由 `ccdl_comm/csrc/` 和 `ccdl_comm/csrc_ascend/` 唯一拥有。

## 5. 删除与保留边界

### 5.1 删除

- 所有受 Git 管理的旧版 Python、CUDA/C++、测试和 benchmark；
- 仅描述旧版工程的根目录文档和构建配置；
- `ccdl_comm_refactor` 内的 `.pytest_cache`、`.pytest_tmp*`、
  `.ruff_cache`、`__pycache__`、egg-info、build 目录和未跟踪
  `.superpowers` 临时目录。

### 5.2 保留

- `.git/` 及全部历史；
- 根目录 `.gitattributes`；
- 重构版的全部源码、测试、报告、需求、设计、计划和贡献规范；
- 当前 `origin` 远程和 `wj_dev` 分支。

忽略规则将改为根目录相对路径，不再包含
`ccdl_comm_refactor/.pytest_tmp*` 等失效前缀。

## 6. 路径与构建调整

代码中的基于 `Path(__file__)` 计算仓库根目录的逻辑在扁平化后仍应满足：

- `packages/<distribution>/setup.py` 的 `PACKAGE_ROOT.parents[1]` 指向仓库
  根目录；
- `ccdl_comm.build.distributions._repository_from()` 返回仓库根目录；
- `tests/packaging/wheel_helpers.py` 从测试目录定位根目录；
- CUDA 和 CANN 源码分别从 `ccdl_comm/csrc` 和
  `ccdl_comm/csrc_ascend` 构建。

文档、测试命令和报告中的有效工作路径应去掉
`ccdl_comm_refactor/` 前缀。历史报告中作为事实记录的远端绝对路径可以保留，
但不得作为当前复现命令。

## 7. 测试设计

迁移采用结构门禁测试驱动：

1. 在迁移前增加布局测试，断言仓库根目录直接拥有 `ccdl_comm/`、
   `packages/`、`tests/` 和重构版 `pyproject.toml`；该测试在当前结构下必须
   失败。
2. 执行删除和移动后让布局测试通过，并断言旧版目录不存在。
3. 运行 Ruff、`git diff --check`、打包专项和完整本地测试。
4. 构建 core wheel，检查 wheel 只包含一次 `ccdl_comm` Python 源码。
5. 在 A6000 的全新远端工作目录验证根目录安装、CUDA extension 加载、
   INT8 量化往返和完整测试。

任何测试不得通过把旧版路径加入 `PYTHONPATH` 来掩盖布局错误。

## 8. 提交与回滚

设计文档单独提交。实际迁移作为一个原子结构提交完成，因为“删除旧根目录”
和“提升重构版”任一单独提交都会留下不可构建状态。

迁移提交遵循：

```text
refactor(repo): flatten lowbit_comm project layout
```

若验证失败，修复应在提交前完成；不得使用 `git reset --hard` 或覆盖用户无关
修改。迁移完成后推送 `wj_dev`，不直接修改受保护主分支。

## 9. 验收标准

- `git ls-files` 中不存在 `ccdl_comm_refactor/`、旧版 `ccdl/`、根级
  `csrc/` 和旧版 `benchmarks/`；
- 根目录项目名称为 `lowbit_comm`，Python 导入仍为 `ccdl_comm`；
- Core/CUDA/Ascend 多包源码所有权保持 Task 18 约束；
- 本地完整测试和打包测试通过；
- A6000 CUDA 扩展可以从根目录干净构建或安装并完成真实量化往返；
- 远程 `wj_dev` 与本地迁移提交一致。
