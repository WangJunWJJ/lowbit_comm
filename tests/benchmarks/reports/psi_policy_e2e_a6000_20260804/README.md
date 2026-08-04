# PSI Policy × lowbit_comm 端到端训练验证报告

## 1. 结论

用户提供的策略训练代码、21 GB Zarr 数据集与 `lowbit_comm` 已完成真实 A6000 单机 2/4 卡端到端联调。数据解析、模型构建、FP16 前后向、INT8 梯度量化通信、验证、checkpoint 保存及 checkpoint 恢复均能执行。

本次一轮训练中，`lowbit_comm` compressed reduce-scatter **没有获得端到端性能收益**：

- 2 卡全局吞吐为 182.16 samples/s，原生 DDP 为 189.95 samples/s，慢 4.10%。
- 4 卡全局吞吐为 342.89 samples/s，原生 DDP 为 375.66 samples/s，慢 8.72%。
- 2→4 卡扩展效率：DDP 为 1.978 倍，CCDL 为 1.882 倍。
- 数据等待只占步骤时间的 1.5%–1.7%，性能差距主要来自量化、残差和 compressed reduce-scatter 路径，而非数据读取。

训练数值保持有限：四组运行均无 NaN/Inf、rank 失步、NCCL/CUDA 错误或策略回退；相同卡数下前 100 步平均损失接近。不过只训练了一轮且每个配置只正式运行一次，当前结果只能证明数值可训练性，不能证明最终任务精度或长期收敛等价。

## 2. 测试对象

| 项目 | 内容 |
|---|---|
| 用户代码 | `归档.zip` 中的 `psi_policy` v2 训练代码 |
| 数据集归档 | `pis-policy-v1-align-10015_open-paper-bag_C8JXLG_20260730_101431.tar`，21,583,206,400 bytes |
| 数据格式 | 4 个 Zarr v3 volume；三路 RGB + 腰/臂/左右手动作 |
| 训练/验证样本 | 44,899 / 2,266；分布式 sampler 补齐为 44,928 个训练样本 |
| 模型 | 三视角 diffusion policy，44,956,124 参数，随机初始化 |
| lowbit_comm | commit `754b8b46a31c41020058d33c7ae1d27f4dfabcda` |
| 机器 | 单机 NVIDIA A6000，2 卡与 4 卡 |
| 容器 | `psi-policy-ccdl:a6000-cu126-torch25` |

## 3. 统一测试口径

- FP16 Accelerate/DDP，每 rank batch size 16。
- 训练 1 个完整 epoch，并执行完整 validation。
- 图像预训练权重关闭，`torch.compile` 关闭，数据 cache 关闭。
- dataloader 为 4 workers/rank、persistent workers。
- CCDL 使用 INT8、group size 64、error feedback、`min_compress_numel=4096`、compressed reduce-scatter/full-gradient restore。
- 性能统计丢弃前 20 个 warm-up step；吞吐使用剩余步骤总耗时计算。
- 每个正式配置为一次独立完整运行，不是多次运行中位数。
- checkpoint 均成功生成，单份大小 540,107,880 bytes。

各目录的 `.hydra/overrides.yaml` 保存了实际命令覆盖项。完整 Hydra 配置未归档，以避免复制用户工程中的外部服务凭据。

## 4. 性能结果

| 配置 | 步数 | 吞吐 (samples/s) | 中位 step (ms) | P95 step (ms) | 峰值显存 (MiB) | 平均 GPU 利用率 |
|---|---:|---:|---:|---:|---:|---:|
| 2 GPU DDP | 1404 | 189.95 | 166.24 | 180.72 | 6231 | 48.92% |
| 2 GPU CCDL RS | 1404 | 182.16 | 174.45 | 184.20 | 6549 | 50.38% |
| 4 GPU DDP | 702 | 375.66 | 168.75 | 179.48 | 6251 | 36.77% |
| 4 GPU CCDL RS | 702 | 342.89 | 183.00 | 215.82 | 6421 | 39.42% |

相对原生 DDP：

| 配置 | 吞吐变化 | 峰值显存变化 | 数据等待占比 |
|---|---:|---:|---:|
| 2 GPU CCDL RS | -4.10% | +318 MiB / +5.10% | 1.58% |
| 4 GPU CCDL RS | -8.72% | +170 MiB / +2.72% | 1.53% |

4 卡的退化更大，且 P95 尾延迟明显升高，说明当前 full-gradient restore、bucket 级量化/残差处理或 collective 调度成本会随 rank 数增加；在单机 A6000 的高速互联上，节省的通信字节不足以抵消这些固定和扩展性开销。

## 5. 训练损失与数值行为

| 配置 | 前 100 步均值 | 后 100 步均值 | epoch train loss | validation loss |
|---|---:|---:|---:|---:|
| 2 GPU DDP | 8.43755 | 2.65207 | 3.84677 | 1.98528 |
| 2 GPU CCDL RS | 8.43605 | 2.90012 | 3.94787 | 2.10885 |
| 4 GPU DDP | 7.53908 | 3.05955 | 4.17093 | 3.04986 |
| 4 GPU CCDL RS | 7.53266 | 3.06679 | 4.18863 | 3.01935 |

2 卡 CCDL validation loss 比 DDP 高 6.22%，4 卡低 1.00%。这个方向不一致，且只有单 seed、单 epoch，因此不能把差异解释为精度提升或确定性精度损失。可以确认的是：量化训练没有立刻发散，首段损失与 DDP 基线高度接近，完整 epoch 可下降并完成验证。

## 6. 功能闭环与发现的问题

### 已验证

- 单 GPU 真实数据 smoke：50 step 训练、完整小规模验证、checkpoint 保存。
- 2 GPU DDP 与 CCDL smoke；4 GPU CCDL batch-size 16 smoke。
- 2/4 GPU 全量一轮训练、完整 validation、checkpoint 保存。
- 4 GPU 从 CCDL checkpoint 恢复模型、优化器和 global step，通信 hook 恢复后继续训练。
- 运行日志确认请求与生效策略均为 `reduce_scatter`，未发生 fallback。
- CCDL 接入适配器的 5 项聚焦测试通过。

### 闭环或隔离的问题

1. 用户代码的 `action_space` 包会在 import 时强制加载未随归档提供的可选 `utils.transforms`。测试副本改成惰性可选 import 后，默认 pass-through 数据路径恢复正常；这是用户代码打包完整性问题，不是 CCDL 故障。
2. 用户代码配置中存在硬编码的外部通知凭据。测试强制关闭通知，报告不保存该值；建议立即轮换凭据，并改用环境变量或 secret manager。
3. 用户代码的 checkpoint 写入 `epoch=0` 后，resume 会再次执行 epoch 0，而不是从下一 epoch 开始。模型、优化器和 global step 能恢复，但 epoch cursor 语义需要在训练代码中修复。
4. 用户代码触发 PyTorch `torch.load(weights_only=False)` 安全告警。可信自有 checkpoint 可运行，外部 checkpoint 应切换到安全加载策略。
5. 原始归档没有 requirements、`pyproject.toml` 或安装脚本；本次根据 import 依赖构建了派生镜像，建议补充锁定依赖与容器清单。

## 7. 性能判断与下一步

当前路径不应作为该模型的默认通信策略。性能优先级建议如下：

1. 对真实 bucket 数量、shape、量化 kernel、EF update、full restore 和 NCCL 时间做 CUDA profiler 分解，先量化每项开销。
2. 提高 `min_compress_numel` 并按 bucket 大小选择 DDP/压缩策略，避免小张量量化成本超过通信收益。
3. 将 quant-pack、reduced-output、dequant/mean/EF update 做 bucket-level CUDA 融合并复用 workspace，降低 launch、allocator 和 callback 开销。
4. 对 sharded consumer 直接输出 `ReducedShard`，移除恢复完整梯度的 final all-gather；DDP consumer 保留安全 fallback。
5. 在单机和双机分别做 3 次以上重复测试。跨机低带宽场景更可能体现压缩收益，但不能用 synthetic benchmark 替代真实策略模型结果。
6. 至少训练 3 个 seed、多个 epoch，并使用实际机器人任务指标或离线 action metric，才能形成精度/收敛结论。

## 8. 报告文件

- `formal_summary.json`：四组正式运行的结构化汇总。
- `formal_*/logs.json.txt`：逐 step 结构化训练事件。
- `formal_*/gpu_samples.csv`：GPU 利用率与显存采样。
- `formal_*/console.log`、`train.log`：完整控制台与训练日志。
- `resume_4gpu_ccdl_rs/`：4 卡 checkpoint 恢复验证日志。
