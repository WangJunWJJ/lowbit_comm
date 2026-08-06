# PSI Policy fused compressed restore 验证

## 结论

本轮专用融合路径已通过 A6000 单机 2/4 卡微基准和 21 GB 真实数据完整训练。相对融合前的 INT8 ReducedShard restore 历史同口径结果，2 卡训练吞吐提升 **4.91%**，4 卡提升 **9.34%**；两个配置均完成完整 1 epoch、全量 validation 和 checkpoint。

| 配置 | 融合前吞吐 | 融合后吞吐 | 提升 | P50 step | P95 step | train loss | val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 GPU | 199.183 samples/s | 208.971 samples/s | +4.91% | 151.235 ms | 162.011 ms | 3.945542 | 2.145954 |
| 4 GPU | 364.307 samples/s | 398.317 samples/s | +9.34% | 160.756 ms | 167.644 ms | 4.164430 | 2.858611 |

所有吞吐统计均丢弃前 20 step。4 卡相对 2 卡的吞吐扩展比为 1.906×。

## 实现与根因

首版融合 kernel 的 CUDA event 隔离测试表明，`dequant-reduce-mean + requantize` 和 gathered payload 单次反量化本身均快于旧算子链；端到端最初回退来自通用 stream-safe workspace pool 的租约、key、event 和 Python 调度成本。

最终保留两种明确的所有权路径：

- 异步或多 stream 场景继续使用通用 `CudaShardWorkspaceProvider`，由 event 保护生命周期。
- 明确有序的同步热路径可注入 caller-owned requantized/gather workspace allocator；调用方负责不与在途异步调用别名。

该方案没有扩大 kernel 支持矩阵。融合 kernel 仍只处理已经验证的 linear INT8、group size 64、top-k 0、non-compact、1–8 payload 快路径；其余配置安全返回 `False` 并执行既有 fallback。

## 微基准

固定 4,194,304 元素，30 warmup、300 iterations、3 个轮换顺序 trial，取中位数；4 卡严格使用 GPU 1、2、3、4。

| 配置 | FP16 restore | 旧 compressed | fused direct workspace | vs FP16 | vs 旧 compressed |
|---|---:|---:|---:|---:|---:|
| 2 GPU | 0.7267 ms | 0.6379 ms | 0.5600 ms | 1.298× | 1.139× |
| 4 GPU | 1.4971 ms | 1.1635 ms | 1.1444 ms | 1.308× | 1.017× |

数值验证：融合输出与旧 compressed 输出逐元素相同，rank 间最大差异为 0；相对精确 FP all-reduce 的 relative L2 约 0.84%。两个 restore workspace 在稳态总计只分配 2 次。

## 真实训练口径

- 数据集：`pis-policy-v1-align-10015/open-paper-bag_C8JXLG`，归档 21,583,206,400 bytes，解压约 21 GB。
- 模型：三视角 diffusion policy，44,956,124 参数，AdamW，batch size 16/rank，1 epoch。
- 日志实际显示 `Mixed Precision: no`，因此本轮真实训练是 FP32 计算/梯度配合 INT8 通信；微基准另按 FP16 restore 口径执行。
- 2 卡使用 GPU 1、2；4 卡使用 GPU 1、2、3、4。
- 两次运行分别完成 1404/702 step，validation 和 540,105,768-byte checkpoint。

2 卡 loss 相对融合前单次运行略高，4 卡略低，方向并不一致。单 seed、单 epoch 只能证明没有发散、NaN、rank 不一致或收敛步数增加，不能证明最终任务精度等价或提升。

## 限制

- 每个训练配置只有一次完整运行，性能比较引用同日、同模型、同数据和同 GPU 集合的历史融合前结果。
- 未执行多 seed 和机器人下游任务指标。
- caller-owned 轻量 workspace 只适用于调用方能保证有序复用的路径；自动异步安全仍应使用通用 pool。

完整结构化数据见 [summary.json](summary.json)。
