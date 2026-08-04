# Task 17：完整 native collective 协议验证

## 结论

Task 17 已完成。CUDA backend 现在通过统一的编译协议支持以下九种 native
Torch/NCCL collective：`all_reduce`、`all_gather`、`reduce_scatter`、
`all_to_all`、`broadcast`、`reduce`、`gather`、`scatter` 和 `barrier`。

2 卡与 4 卡 A6000 对同步、异步两种模式逐项验证，所有结果均与解析参考值一致，
跨 rank 最大绝对误差为 0。未实现的压缩 collective 在编译阶段抛出
`UnsupportedCollective`；只有调用方显式声明的 fallback 才能回退，并写入
`ExecutionInfo`。

## 测试环境

- 主机：5 × NVIDIA RTX A6000（本任务使用 2 卡和 4 卡）
- 容器：`ccdl-comm-a6000:cu126-torch25`
- Torch：`2.5.0a0+872d972e41.nv24.08`
- 通信 backend：NCCL
- 验证 commit：`eda2d4e`

## 测试命令

```bash
python -m pytest tests/conformance -q
CCDL_COMM_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation
python -m pytest tests -q
torchrun --standalone --nproc-per-node=2 \
  tests/distributed/native_collective_smoke.py \
  --output-json tests/benchmarks/reports/task17_native_collectives/raw_2gpu.json
torchrun --standalone --nproc-per-node=4 \
  tests/distributed/native_collective_smoke.py \
  --output-json tests/benchmarks/reports/task17_native_collectives/raw_4gpu.json
```

源码挂载但尚未 editable install 时，分布式命令需设置仓库根目录到
`PYTHONPATH`；安装后无需该环境变量。

## 结果

| 验证项 | 结果 |
|---|---:|
| 本地 conformance | 52 passed |
| 本地全量回归 | 733 passed, 30 skipped |
| A6000 conformance | 52 passed |
| A6000 CUDA 扩展全量回归 | 879 passed, 1 skipped |
| A6000 2 卡 native smoke | 9/9 collective，sync/async，误差 0 |
| A6000 4 卡 native smoke | 9/9 collective，sync/async，误差 0 |

原始结果见 [raw_2gpu.json](raw_2gpu.json) 和
[raw_4gpu.json](raw_4gpu.json)。

## 测试中发现并闭环的问题

1. `collectives.__init__` 直接导出同名快捷函数会遮蔽历史子模块导入。最终将九种
   快捷入口放在顶层 `ccdl_comm` 和 `ccdl_comm.collectives.api`，保留原子模块语义。
2. 首次远端全量测试未构建 `ccdl_cuda_ops`，125 个 CUDA case 在 fixture 阶段
   报错。使用标准开关 `CCDL_COMM_BUILD_CUDA=1` 构建后，全量回归通过。
3. A6000 到 GitHub 的一次 clone 因 TLS 中断失败；测试改用相同 commit 的 Git
   bundle 部署，不影响代码或结果口径。

本 smoke 只验证协议正确性、Work 完成语义和 rank 一致性，不宣称 native
collective 的吞吐性能提升；性能基准仍应使用已有的专用 benchmark。
