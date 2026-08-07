# 4 卡 sharded parameter pipeline：21 GB 真实数据验证

## 结论

在单机 4 张 RTX A6000（固定物理 GPU 1、2、3、4）上，使用 21 GB PSI Policy
真实数据、约 4496 万参数模型、FP16、AdamW 和每 rank batch 16，四种模式各交替执行
3 个完整 epoch。每次运行均包含 702 个训练 step、36 个验证 step 和标准 checkpoint。

`sharded_compressed` 的三次吞吐中位数为 **380.260 samples/s**，相对 Native DDP
提升 **7.57%**，相对现有 `full_fused` 提升 **0.47%**。它通过了性能下限、rank
一致性、有限值、workspace 稳定和无 fallback 门禁，但验证损失为 **4.704630**，相对
`sharded_fp` 的 **2.938473** 高 **60.10%**。因此该路径只证明了性能潜力，当前必须
保持显式 opt-in，不能设为默认策略。当前 4 卡生产推荐仍应使用 `full_fused`。

## 严格同口径性能结果

统计统一丢弃每次运行前 20 个训练 step，以全局 batch 64 除以余下 682 个 step 的
平均耗时计算吞吐，再取三次中位数。

| 模式 | 三次吞吐（samples/s） | 中位数 | vs Native | vs full fused |
|---|---|---:|---:|---:|
| Native DDP | 356.764 / 352.530 / 353.511 | **353.511** | 基线 | -6.59% |
| full fused CCDL | 387.410 / 378.464 / 374.268 | **378.464** | **+7.06%** | 基线 |
| sharded FP restore | 355.915 / 335.969 / 353.050 | **353.050** | -0.13% | -6.72% |
| sharded compressed restore | 377.049 / 380.260 / 387.093 | **380.260** | **+7.57%** | +0.47% |

| 模式 | P50 step 中位数 | P95 step 中位数 | epoch train loss | val loss |
|---|---:|---:|---:|---:|
| Native DDP | 176.631 ms | 204.046 ms | 4.184546 | 2.982103 |
| full fused CCDL | 166.919 ms | **182.492 ms** | 4.187568 | 2.945411 |
| sharded FP restore | 179.972 ms | 191.808 ms | 4.214159 | **2.938473** |
| sharded compressed restore | **163.684 ms** | 195.788 ms | 6.660785 | 4.704630 |

compressed 路径的 P50 最低，但 P95 高于 full fused，且仅有 0.47% 的中位吞吐优势，
尚不足以证明它在性能上显著优于 full fused。相反，训练效果退化在三次运行中完全一致，
是确定性的路径语义问题，而不是运行波动。

## 正确性和工程门禁

- 12/12 运行均有 `.complete`、结构化日志和 checkpoint，且明确记录 FP16。
- 所有 loss 均为有限值，未出现 traceback、CUDA error、NCCL warning 或 fallback。
- 两种 sharded 路径的最大跨 rank 参数差均为 `0.0`，workspace 指针保持稳定。
- 每次运行有 702 个全局 step；AMP 检出 2 次 overflow，所有 rank 一致跳过，因此执行
  700 次 AdamW 更新，没有发生 rank 步数分歧。
- 每 rank 的 AdamW 分片状态为 22,478,848 个元素；checkpoint 保存时物化为标准完整
  optimizer state。sharded checkpoint 为 540,040,985 bytes（约 515.02 MiB）。
- 最终 A6000 CUDA 回归为 **157 passed, 1 skipped**。

有限值和 rank 一致并不等同于训练效果正确。报告采用“相对 `sharded_fp` 的验证损失增幅
不超过 5%”作为保守门禁；实际增幅为 60.10%，远超任何合理的数值噪声范围，所以总门禁
明确失败。

## 问题根因

`sharded_fp` 证明 rank-local AdamW、compressed reduce-scatter 和参数分片更新本身没有
造成明显 loss 退化。退化只在 `sharded_compressed` 出现，差异点是：每个 rank 更新本地
FP 参数分片后，将参数再次量化为 INT8 做 all-gather，各 rank 在下一次前向计算前只恢复
到反量化后的 FP16 近似权重。这个过程每一步都对完整计算权重重新施加参数量化误差；梯度
通信上的 error feedback 不会补偿参数恢复误差，因此误差持续进入前向、反向和后续 AdamW
轨迹。结果是吞吐提高，但优化目标被明显扰动。

这也解释了为什么“INT8 一直传到计算前再反量化”在通信量上合理，却不能仅凭 rank 一致性
保证收敛：所有 rank 可以完全一致地使用同一组有偏近似权重。

## 阶段耗时与扩展性说明

真实 PSI 日志提供端到端 step time，但没有在训练 step 内同步插入五阶段计时，避免计时事件
改变 overlap 行为。因此本报告不伪造真实阶段耗时。Task 7 的独立 4 卡 synthetic 同版本参考为：

| 阶段 | 中位耗时 |
|---|---:|
| backward flatten | 1.8449 ms |
| compressed reduce-scatter | 6.8126 ms |
| rank-local AdamW update | 0.1053 ms |
| parameter quantize + gather | 0.1030 ms |
| parameter restore writeback | 0.0034 ms |

这些值只用于定位算子级方向，不能与真实 PSI step 直接相加。真实报告中的
`process_hwm_mib` 是 CPU 进程高水位，不是 GPU 峰值，因此没有将其误标为显存指标。

历史严格 FP16 矩阵可提供 Native/full fused 的 2 卡结果，但使用了不同 CUDA 扩展哈希；
据此计算的 4/2 吞吐比分别约 1.791x 和 1.848x，仅作背景参考。新 sharded 路径没有同版本
2 卡真实数据结果，故不宣称其 4/2 扩展效率。

## 下一步优化建议

1. **先修训练效果，再继续默认化。** 为参数通信单独引入 FP master shard 与参数残差，采用
   error-feedback/stochastic rounding；增加周期性 FP16 全量刷新，并对敏感张量使用 FP16
   或更高 bit。每个方案必须先通过多 epoch loss/任务指标门禁。
2. **把参数恢复从 flat 全局屏障改为 bucket/layer 流水。** 让下一层所需参数的 INT8 gather、
   反量化与当前层计算重叠；当前一次性恢复完整参数限制了 overlap 上限。
3. **融合本地更新链。** 将 reduced-shard dequant、AdamW、参数 requant 和残差更新合入
   更少的 kernel，并继续复用 send/recv/output workspace，降低 P95 抖动。
4. **对 0.47% 增益保持谨慎。** 在训练效果修复后至少再做 5 次交替运行，并加入 CUDA event
   分阶段计时和 GPU 峰值采样，确认相对 full fused 的收益超过系统噪声。
5. 若消费者能直接执行真正的参数/计算分片，应避免每 step 恢复完整参数；否则 full parameter
   all-gather 的拓扑成本仍会限制 8/64 卡扩展性。

首次 Native 运行额外生成共享 normalizer cache，因此墙钟为 369 秒，其余约 187--199 秒；
该冷启动时间不计入训练 step 吞吐。完整结构化汇总见 [summary.json](summary.json)，12 次紧凑
原始结果见 [raw/runs.json](raw/runs.json)。
