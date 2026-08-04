# lowbit_comm 根布局 A6000 验证报告

## 结论

提交 `62ba510d3e0be44a9b1c695964e4dc1496df0840` 的仓库扁平化验证通过。
远程仓库根目录直接包含 `ccdl_comm/`、`packages/`、`tests/` 和 `docs/`，
不存在旧版 `ccdl/`、`csrc/`、`benchmarks/` 或暂存目录
`ccdl_comm_refactor/`。

## 环境与产物

- 主机：`user-SYS-6049GP-TRT-LongJing-Server`
- GPU：NVIDIA RTX A6000
- 驱动：550.142
- 容器：`ccdl-comm-a6000:cu126-torch25`
- PyTorch：`2.5.0a0+872d972e41.nv24.08`
- CUDA：12.6
- Core wheel：`ccdl_core-0.1.0-py3-none-any.whl`，约 136 KiB
- CUDA wheel：`ccdl_cuda-0.1.0-cp310-cp310-linux_x86_64.whl`，约 17 MiB

## 验证结果

- Core 和 CUDA wheel 均从扁平化仓库根布局成功构建。
- 两个 wheel 在临时目录隔离安装成功，`ccdl_cuda_ops` 可加载。
- FP16 输入经过 INT8 量化/反量化后的最大绝对误差为 `0.0078125`，
  未劣于 Task 18 基线。
- A6000 全量测试：`893 passed, 1 skipped`。
- 双卡 compiled all-reduce smoke：通过；`shortcut_max_abs_difference=0.0`，
  相对 L2 误差为 `0.00595319`。本次短 smoke 的 compiled 路径为
  `0.3460 ms`，原生 FP16 all-reduce 为 `0.3506 ms`；该数值只用于迁移
  正确性和明显性能退化检查，不替代正式 benchmark。

主要复现命令：

```bash
python -m pip wheel --no-build-isolation --no-deps \
  -w dist/a6000 packages/ccdl-core
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
  python -m pip wheel --no-build-isolation --no-deps \
  -w dist/a6000 packages/ccdl-cuda
PYTHONPATH=/workspace/.pytest_tmp_a6000_site:/workspace python -m pytest -q
PYTHONPATH=/workspace/.pytest_tmp_a6000_site:/workspace \
  torchrun --standalone --nproc-per-node=2 \
  tests/distributed/cuda_compiled_backend_smoke.py \
  --numel 1048576 --warmup 2 --repeat 5
```

## 测试中发现并闭环的问题

1. 首次直接跑 GPU 全量测试时，125 个 CUDA 用例因容器尚未安装
   `ccdl_cuda_ops` 而报错。先构建并隔离安装 Core/CUDA wheel 后重跑，
   相关用例全部通过；这是验证顺序问题，不是源码缺陷。
2. 镜像未安装 `python-build`，因此文档中的 `python -m build` 无法执行。
   改用已有的 `pip wheel --no-build-isolation --no-deps` 调用同一
   setuptools 构建后端，避免联网改变 Torch/CUDA 环境，构建成功。
3. SSH 命令输出存在缓冲，不影响执行结果；未并发启动重复构建，避免
   产物竞争和资源污染。

结构化原始结果见
[`raw_a6000_layout.json`](raw_a6000_layout.json)。
