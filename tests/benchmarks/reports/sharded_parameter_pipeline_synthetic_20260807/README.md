# Sharded parameter pipeline synthetic gate — 2026-08-07

## 结论

在同一 A6000 主机、容器、扩展、源码、FP16 模型和 batch 口径下，新的
`sharded_compressed` 路径通过 2 卡和 4 卡性能 gate：

| 模式 | 2 卡中位吞吐（samples/s） | 4 卡中位吞吐（samples/s） |
|---|---:|---:|
| Native DDP | 3007.71 | 3031.65 |
| full_fused | 3485.77 | 2594.75 |
| sharded_fp | 3304.12 | 3399.52 |
| sharded_compressed | 4102.92 | 4492.55 |

- 2 卡 `sharded_compressed` 相对 Native DDP：`+36.41%`；相对 `full_fused`：`+17.70%`。
- 4 卡 `sharded_compressed` 相对 Native DDP：`+48.19%`；相对 `full_fused`：`+73.14%`。
- 24 个正式运行全部 loss 有限、rank 最大参数差异为 0；compressed 路径无 fallback，workspace 指针稳定。

这些数据证明 synthetic 约 4495 万参数模型上的初始性能门槛已经通过，但不能替代 21 GB
真实数据集的训练精度、验证集指标和完整 epoch 结论。

## 固定口径

- 机器：`user@192.168.8.156`，NVIDIA RTX A6000。
- 2 卡物理编号：`1,2`；4 卡物理编号：`1,2,3,4`。
- 镜像 ID：`sha256:24821ea669cced538c5b0b2f5a8e15ce61700003a49daf259bd42406bbd70693`。
- 源码：`1c6228d`；CUDA 扩展 SHA-256：`3a1186ed...304dd46`。
- FP16，约 4495 万实际参数，batch 16/rank，50 步 warmup，200 步计量。
- 每模式 3 次，并按不同模式顺序轮换；聚合使用中位数而非最好值。

## 观察与限制

4 卡使用的 GPU 1/2 位于一个 PIX 域，GPU 3/4 位于另一个 NODE 域，两组之间为 SYS。
`compressed_reduce_scatter` 的阶段中位时间由 2 卡约 2.93 ms 增至 4 卡约 6.81 ms，说明跨
NUMA/SYS 拓扑仍是下一轮优化重点。尽管如此，参数 shard 变小和 INT8 参数恢复使 4 卡端到端
吞吐仍高于其他三种模式。

阶段计时使用 CUDA event，适合比较当前流上的算子提交和 kernel，但不能完整表示异步 NCCL
在独立通信流上的暴露时长。因此性能 gate 以包含逐步同步的端到端吞吐为准，阶段数据只用于
定位方向，不用于相加推导总 step latency。

原始 JSON 与日志保留在远端：

- `/home/user/wangjun/lowbit_comm_task6_pipeline/results/task7_2gpu_final`
- `/home/user/wangjun/lowbit_comm_task6_pipeline/results/task7_4gpu_final`

结构化汇总见 `summary.json`。
