# Task 18 多包构建与安装验证报告

## 结论

Task 18 通过。发布边界为：

- `ccdl-core`：唯一拥有 `ccdl_comm` Python 源码，不依赖 Torch；
- `ccdl-cuda`：只包含 `ccdl_cuda_ops` 原生扩展，依赖
  `ccdl-core==0.1.0` 和 Torch；
- `ccdl-ascend`：只包含 `ccdl_cann_ops` 原生扩展，依赖
  `ccdl-core==0.1.0`、Torch 和 torch-npu；
- `ccdl-comm`：不拥有源码的兼容性元包。

三种发行物共享仓库中的唯一实现，构建过程不复制 Python、CUDA 或 CANN
源码。Core ABI 当前为 `CCDL_CORE_ABI = 1`。

## 验证矩阵

| 环境 | 验证内容 | 结果 |
| --- | --- | --- |
| Windows/Python | wheel 所有权、元数据、core-only import、缺失扩展诊断 | 12 passed |
| 单机 A6000/CUDA | Core 与 CUDA 独立构建、隔离安装、真实扩展量化/反量化 | passed |
| Ascend/CANN | Core 与 Ascend 独立构建、隔离安装、扩展加载及算子烟雾测试 | passed |
| A6000 全量回归 | 项目测试集 | 891 passed, 1 skipped |

原生 wheel 信息：

- CUDA：`ccdl_cuda-0.1.0-cp310-cp310-linux_x86_64.whl`，17,556,485 bytes；
- Ascend：`ccdl_ascend-0.1.0-cp311-cp311-linux_aarch64.whl`，4,426,323 bytes；
- Core：纯 Python wheel，约 138 KB（不同构建环境 ZIP 元数据存在少量差异）。

CUDA 与 CANN 扩展的 INT8 量化/反量化往返最大绝对误差均为
`0.0078125`。原始结果分别保存在
`raw_a6000_cuda_wheel.json` 和 `raw_ascend_cann_wheel.json`。

> Ascend 本项验证聚焦发行边界：原生扩展完成构建、隔离安装和动态加载，
> 并执行 CPU 张量兼容算子路径。NPU 设备侧性能不属于 Task 18 门禁。

## 复现命令

在目标 Torch 环境中关闭构建隔离，避免构建系统从网络解析出另一套
Torch/CUDA/CANN：

```bash
python -m build --wheel --no-isolation packages/ccdl-core
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
  python -m build --wheel --no-isolation packages/ccdl-cuda
CCDL_COMM_BUILD_CANN=1 MAX_JOBS=2 \
  python -m build --wheel --no-isolation packages/ccdl-ascend
```

安装矩阵使用临时 `--target` 目录验证，以排除源码树和系统 site-packages
污染。常规安装时将 Core wheel 与且仅与一个目标后端 wheel 一并安装。

## 测试中发现并闭环的问题

1. 后端 wheel 不拥有 Python 包时，Torch `BuildExtension` 未提前创建 Ninja
   临时目录。现由安全 build-ext 包装器初始化 workspace。
2. Python 3.10 没有 `tomllib`。打包测试使用 `tomli` 兼容回退。
3. wheel metadata 会规范化包名。断言改为按 PEP 508 解析，而非比较原始字符串。
4. 源码目录和 `PYTHONPATH` 会掩盖隔离安装错误。安装测试从临时工作目录启动
   子进程并清理继承路径。
5. 远端默认索引不可达。构建工具离线注入；原生构建使用目标环境现有 Torch。

CUDA 编译仍会显示既有 C++ 可见性与 CUDA 未使用变量告警；它们不影响本次
构建和运行结果，也未在 Task 18 中进行无关 ABI 改写。
