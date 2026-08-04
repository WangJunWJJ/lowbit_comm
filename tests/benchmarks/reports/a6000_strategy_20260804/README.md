# A6000 多 rank 压缩通信策略性能报告

## 结论

4/8 rank 不应继续默认使用全量量化 `all-gather`。本轮真机结果支持按拓扑选择策略：

- 单机 4 卡：`compressed reduce-scatter + FP16 full restore` 吞吐为 DDP 的 `1.099x`，比旧 CCDL all-gather 快 `1.338x`。
- 双机 4 卡：hierarchical 吞吐为 DDP 的 `1.464x`，比旧 CCDL all-gather 快 `1.509x`，是当前最佳完整梯度路径。
- 双机 8 卡：hierarchical 比旧 CCDL all-gather 快 `3.436x`，但仍比 DDP 慢约 `4.2%`；compressed reduce-scatter 虽比旧 all-gather 快 `1.660x`，但最终 FP16 full all-gather 抵消了大部分压缩收益。
- ReducedShard consumer 不恢复完整梯度时，通信级延迟相对 full restore 在单机 4 卡、双机 4 卡、双机 8 卡分别降低到 `1/2.384`、`1/2.754`、`1/2.159`。这证明 sharded consumer 是 4/8 rank 下一阶段的关键路径，但当前数据不是端到端优化器训练结果。

因此建议默认策略为：单机 4 卡优先 compressed reduce-scatter；当前 1GbE 双机 4/8 卡优先 hierarchical；拥有分片优化器或 FSDP-style consumer 时直接消费 `ReducedShard`，不要执行最终完整梯度 all-gather。策略仍须 capability-gated，未知拓扑保持安全 fallback。

## 测试口径

- 源码提交：`754b8b46a31c41020058d33c7ae1d27f4dfabcda`
- 镜像：`ccdl-comm-a6000:cu126-torch25`
- GPU：NVIDIA RTX A6000
- 完整训练：FP16 synthetic MLP，`62,914,560` 参数，batch size 每 rank 16，SGD
- 每次训练 20 步，前 5 步预热不计时；每个配置独立运行 3 次，取中位数
- CCDL：INT8、group size 64、error feedback
- 双机网络：`eno2` 1GbE TCP，显式设置 `NCCL_IB_DISABLE=1`；本报告不代表 RDMA 性能
- baseline 和旧 all-gather 数值复用同源码、同模型、同机器、同口径的 `a6000_scale_20260804` 正式结果

## 端到端完整梯度训练

| 拓扑 | 策略 | 中位 step ms | 中位 samples/s | 相对 DDP | 相对旧 all-gather |
| --- | --- | ---: | ---: | ---: | ---: |
| 单机 4 卡 | PyTorch DDP FP16 | 28.585 | 2238.94 | 1.000x | — |
| 单机 4 卡 | CCDL INT8 all-gather | 34.799 | 1839.14 | 0.821x | 1.000x |
| 单机 4 卡 | CCDL compressed reduce-scatter + full restore | **25.999** | **2461.65** | **1.099x** | **1.338x** |
| 双机 4 卡 | PyTorch DDP FP16 | 1618.775 | 39.54 | 1.000x | — |
| 双机 4 卡 | CCDL INT8 all-gather | 1668.504 | 38.36 | 0.970x | 1.000x |
| 双机 4 卡 | CCDL compressed reduce-scatter + full restore | 1518.596 | 42.14 | 1.066x | 1.099x |
| 双机 4 卡 | CCDL hierarchical | **1105.845** | **57.87** | **1.464x** | **1.509x** |
| 双机 8 卡 | PyTorch DDP FP16 | **1084.083** | **118.07** | **1.000x** | — |
| 双机 8 卡 | CCDL INT8 all-gather | 3886.840 | 32.93 | 0.279x | 1.000x |
| 双机 8 卡 | CCDL compressed reduce-scatter + full restore | 2342.167 | 54.65 | 0.463x | 1.660x |
| 双机 8 卡 | CCDL hierarchical | 1131.224 | 113.15 | 0.958x | **3.436x** |

所有 15 次新正式训练均完成，没有 fallback、CUDA/NCCL 错误或非有限 loss。相同拓扑下，两种新 CCDL 策略记录的 20 步平均 loss 一致：4 rank 为 `0.9972114623`，8 rank 为 `0.9983771570`。这只能证明短程数值 smoke 通过，不能替代真实数据集上的最终精度、收敛步数和长程稳定性验证。

## ReducedShard 通信路径

该测试使用 `16,777,216` 个 FP16 元素，INT8/group 64，启用 compiled chunk plan、workspace cache、caller-owned output 和 fused dequant-reduce。每次预热 10 次、测量 30 次，并以 `full-shard-shard-full` 顺序抑制测量次序偏差；每个拓扑再独立运行 3 次取中位数。

| 拓扑 | full restore ms | ReducedShard ms | 通信级加速 | relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 单机 4 卡 | 6.051 | 2.542 | **2.384x** | 0.005943 |
| 双机 4 卡 | 423.844 | 150.349 | **2.754x** | 0.005943 |
| 双机 8 卡 | 669.545 | 310.815 | **2.159x** | 0.005947 |

9 次正式 ReducedShard 运行均满足：输出指针稳定、rank metadata 数量与 world size 一致、无非有限值、relative L2 小于 `0.02`，且 ReducedShard 延迟低于 full restore。

这里的收益来自直接输出 reduced shard，省去了为 DDP 完整 bucket 恢复全量梯度的最终 FP16 all-gather。只有调用方能直接消费分片梯度时才能兑现；若随后仍拼回完整梯度，收益会重新被吃掉。

## 解释与后续方向

双机 8 卡 compressed reduce-scatter full-restore 仍慢，是因为当前路径完成压缩分片交换后，为兼容 DDP full bucket 又执行一次完整 FP16 all-gather；在 1GbE 和 8 rank 下，这一步成为主导开销。hierarchical 将跨机流量集中到节点级阶段，因此显著优于每 rank 全量收集，但在 8 卡下仍有本地聚合、跨节点交换和广播链开销。

下一步的性能优先级：

1. 为独立 CCDL API 完善 ReducedShard consumer contract、生命周期和异步完成语义，让分片优化器直接消费输出。
2. 将 hierarchical 的跨节点阶段改为真正 compressed reduce-scatter，并避免节点内最终 full restore。
3. 在可用的统一 RDMA fabric 上重测；当前 1GbE 结论只用于相同网络条件下的策略相对比较。
4. 使用固定数据顺序的真实模型进行长程精度与收敛测试，不能从本轮 synthetic 20-step smoke 外推最终精度。

结构化汇总见 [`summary.json`](summary.json)，每次运行的原始证据见 [`raw/`](raw/)；旧 DDP/all-gather 原始对照见相邻的 `a6000_scale_20260804` 报告。
