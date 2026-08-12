# PSI Policy 21G 真实数据端到端验证

## 结论

当前 PR 代码已在 21G Zarr 数据集上完成 2/4 卡真实训练、全量 validation 和 checkpoint 闭环。CCDL compressed reduce-scatter 在 2 卡基本追平 Native DDP（**-0.14%**），4 卡仍慢 **3.31%**。相比 2026-08-04 旧实现的 -4.10%/-8.72%，性能退化已经明显收窄，但本轮不能宣称真实模型获得训练加速。

| 配置 | step | 吞吐（samples/s） | P50 step | P95 step | epoch loss | validation loss |
|---|---:|---:|---:|---:|---:|---:|
| 2 GPU Native DDP | 1404 | 186.254 | 167.795 ms | 198.995 ms | 3.846767 | 1.985282 |
| 2 GPU CCDL RS | 1404 | 185.987 | 170.244 ms | 184.756 ms | 3.947868 | 2.108849 |
| 4 GPU Native DDP | 702 | 352.769 | 179.724 ms | 200.492 ms | 4.170935 | 3.049855 |
| 4 GPU CCDL RS | 702 | 341.109 | 181.423 ms | 222.992 ms | 4.188628 | 3.019351 |

吞吐统计丢弃前 20 step。数据等待占比只有 1.54%–1.64%，因此 CCDL 与 DDP 的差距主要来自通信压缩、error feedback 和 full-gradient 恢复路径，而不是 21G 数据读取。

## 测试对象

- 数据集：`pis-policy-v1-align-10015/open-paper-bag_C8JXLG`，归档 21,583,206,400 bytes，解压后约 21G。
- 样本：44,899 train、2,266 validation；DistributedSampler 补齐为 44,928 train samples。
- 模型：三视角 diffusion policy，44,956,124 参数；图像 encoder 不加载预训练权重。
- 优化器：AdamW；FP16；batch size 16/rank；训练 1 epoch。
- CCDL：INT8、group size 64、error feedback、`min_compress_numel=4096`、compressed reduce-scatter。
- 2 卡使用 GPU0-1；4 卡严格使用 GPU1-4。
- lowbit_comm：`60e514d69934f845341c4adc786dd122d8ff5a0c`。

## 正确性

- 四组均完成全部训练 step、全量 validation 和 checkpoint 保存，checkpoint 均为 540,107,880 bytes。
- 30-step 真实数据 smoke 先行通过。
- PSI Policy CCDL 接入测试为 4 passed。
- CCDL 日志确认 requested/effective strategy 均为 `reduce_scatter`，没有策略 fallback。
- 未发现 Traceback、NaN/Inf、CUDA 或 NCCL 错误。
- 2 卡 CCDL validation loss 相对 DDP 高 6.22%；4 卡低 1.00%。方向不一致，单 seed、单 epoch 不足以判定最终精度损失。

## 为什么没有测试 ReducedShard sharded SGD

真实 PSI Policy 使用 AdamW，而当前 `TorchShardedSgdConsumer` 只支持无 momentum、无 weight decay 的 SGD。强行切换会改变优化器和收敛口径，所得性能与精度对比无效。因此本轮测试的是当前 CCDL 可正确接入 AdamW/DDP 的 full-gradient compressed reduce-scatter 路径。

要把 synthetic MLP 上的 7.46% sharded 收益带到该模型，下一步必须实现并验证 sharded AdamW state（参数、first moment、second moment 的 rank-local ownership）、参数预取和计算重叠；不能只替换通信 hook。

## 限制

- 每个配置只有一次完整运行，尚无三次运行中位数。
- 只训练一个 epoch、一个初始化 seed，不能证明最终收敛等价。
- 没有真实机器人下游任务指标。
- 本轮未启用外部 GPU 显存采样，只保留进程 HWM；不能与 CUDA peak allocated 等价比较。
