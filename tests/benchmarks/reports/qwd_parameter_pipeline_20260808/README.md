# qWD 分片参数通信 A6000 验收报告

## 结论

在单机 4 张 RTX A6000（固定物理 GPU 1、2、3、4）上，使用 21.58 GB PSI Policy
真实数据、44,956,124 参数三视角 diffusion policy、AdamW、FP16 mixed precision 和每
rank batch 16，四种模式按预定顺序交替运行 3 轮，每轮 3 个完整 epoch。共完成 12 次
训练、36 个 epoch、25,272 个训练 step，所有运行均产生 checkpoint 和有限 loss。

`sharded_qwd` 的中位吞吐为 **390.027 samples/s**，相对 Native DDP 的
**365.166 samples/s 提升 6.81%**，相对直接量化完整参数的 `sharded_compressed`
提高 0.70%。其最终验证 loss 为 **2.180156**，低于 `sharded_fp` 的 2.201938，loss
比值 0.9901；因此吞吐、训练效果、rank 一致性、workspace 和 fallback 五类门禁均通过。

qWD 可以作为已验证 4 卡 A6000 拓扑上的 capability-gated 候选默认策略；对其他拓扑、
多机和不同模型仍应保持显式 opt-in，直到完成同口径验收。

## 严格同口径结果

每次运行包含 2,106 个训练 step。统一丢弃最初 20 个 step，以剩余 2,086 个 step 的
全局 batch 64 和平均 step time 计算吞吐，再取三轮中位数。

| 模式 | 三轮吞吐（samples/s） | 中位吞吐 | vs Native | P50 step | P95 step | 最终 val loss |
|---|---|---:|---:|---:|---:|---:|
| Native DDP | 361.581 / 367.099 / 365.166 | **365.166** | 基线 | 173.087 ms | 189.204 ms | **2.140743** |
| sharded FP | 346.730 / 349.140 / 358.498 | **349.140** | -4.39% | 180.207 ms | 196.282 ms | 2.201938 |
| direct INT8 parameter | 387.318 / 386.279 / 388.502 | **387.318** | +6.07% | 162.430 ms | 186.140 ms | 3.787579 |
| INT8 qWD | 390.027 / 393.477 / 388.609 | **390.027** | **+6.81%** | **161.053 ms** | **174.873 ms** | 2.180156 |

qWD 相对 sharded FP 吞吐提高 11.71%。直接 INT8 虽然也快，但最终验证 loss 比 sharded
FP 高 72.01%；qWD 把该退化消除，同时保留了通信压缩的吞吐收益。

## 收敛轨迹

| 模式 | epoch 1 val loss | epoch 2 val loss | epoch 3 val loss |
|---|---:|---:|---:|
| Native DDP | 2.929622 | 2.292976 | 2.140743 |
| sharded FP | 2.965479 | 2.429465 | 2.201938 |
| direct INT8 parameter | 4.299476 | 3.849316 | 3.787579 |
| INT8 qWD | **2.911801** | 2.388308 | **2.180156** |

qWD 三轮的 loss 曲线完全一致。它没有逐步重新量化完整权重，而是保留 FP32 master
shard，传输 `master - model_copy` 的 INT8 差值，并将反量化差值加回 model copy；未恢复
误差自然进入下一步差值，形成隐式 parameter error feedback。

## qWD 策略与通信成本

每轮有 2,106 个全局 step，其中 6 次由 AMP overflow 检测一致跳过，执行 2,100 次
AdamW 更新：2,083 次 fused INT8 qWD，17 次 FP32 refresh。refresh 占有效更新的
0.81%，由 1 次 warmup、4 次周期 refresh 和 12 次误差阈值 refresh 组成。

在相同 4 卡和 44,956,124 参数规模上，单独使用同步 wall-time 测得：

| 参数通信操作 | 中位数 | P95 |
|---|---:|---:|
| fused qWD quantize + INT8 all-gather + add | 5.949 ms | 5.972 ms |
| FP32 shard all-gather refresh | 19.729 ms | 19.801 ms |

qWD 通信耗时约为 FP32 refresh 的 30.15%。该微基准使用 30 个测量迭代和显式 CUDA
synchronize；早先仅依赖 CUDA event 的结果未被采纳，因为 NCCL stream 完成关系会使其
低估 wall time。

误差采样每 128 次有效更新执行一次，与周期 refresh 重合的采样点不重复计入。12 个有效
采样值均为 1.0，超过 0.01 阈值，安全策略在下一步执行 FP32 refresh。该值说明当前全模型
L2 相对误差指标对真实 PSI 参数差值非常保守，后续应继续研究分层/分桶误差度量或 mixed-bit
策略；本次没有调高阈值，也没有删除 refresh 来美化性能结果。

## 正确性和工程门禁

- 2 卡和 4 卡独立 qWD smoke 的最大 FP32 AdamW master oracle 差均为
  `1.4901161193847656e-08`，小于 `1e-6` 门限。
- 2/4 卡 smoke 和三轮真实 qWD 的最大跨 rank 参数差均为 `0.0`。
- qWD 均命中 `fused_int8_qwd`，无 fallback，workspace 指针稳定。
- 四种模式的所有训练和验证 loss 均为有限值。
- 严格门禁：qWD/Native = 1.0681（要求 >=1.05），qWD/direct INT8 = 1.0070
  （要求 >=0.98），qWD/sharded-FP loss = 0.9901（要求 <=1.02）。
- 本地完整回归为 `971 passed, 55 skipped`；A6000 qWD/CUDA/restore 相关回归为
  `164 passed`，另有 2 卡和 4 卡分布式 smoke 各一次。

Native DDP 的跨 rank 一致性由同步 DDP 更新语义确认；当前 PSI runner 没有额外插入参数
all-gather 检查。三种 sharded 路径在训练末尾显式测量了跨 rank 参数差。

## 环境与复现指纹

- CCDL commit：`95d96c211011cfef8a34b0567d8907ae32347606`
- CUDA 扩展 SHA-256：`ee093cdf5030aefe9f9133fda67e0aaa9f9138bfff1f23e432201b4c10406bbc`
- 容器镜像：`sha256:89f988d94bcff81806d95b27522a632b258a578b52587635cb255404ed386335`
- PyTorch/CUDA/NCCL：`2.5.0a0+872d972e41.nv24.08 / 12.6 / 2.22.3`
- 驱动：`550.142`
- 数据集：491 个文件，文件内容总长度 21,582,259,257 bytes；manifest SHA-256：
  `c44ecb3a98fde0b6e983f014f6b1b042615011159993cec812906ba12482e5b9`
- 模型源码 SHA-256：`bd4e0ff9952a0441c6b9bbec5eb89c50d9705f96bc071d527b26005869a231a8`
- 配置 SHA-256：`7cdc79726175234425dbf011326ccae7323d748cc8bd74f8a32107b88b52ee4a`

完整环境字段和 12 次结构化原始结果位于 `raw/`。

## 测量边界

真实 PSI runner 没有注入逐阶段 CUDA synchronize，以免改变 overlap 和端到端吞吐，因此
没有伪造真实训练的阶段耗时。Task 7 的 4 卡 tiny synthetic 参考为：backward flatten
0.266 ms、compressed reduce-scatter 2.994 ms、local AdamW 0.305 ms、parameter delta
quantize 0.239 ms、INT8 all-gather 0.103 ms、add writeback 0.046 ms；这些值只用于定位算子
方向，不能直接与 PSI step 相加。

本次 PSI 日志提供的是 CPU process high-water memory，不是 GPU peak allocated memory；
中位数分别为 Native 2,420.9 MiB、sharded FP 2,739.7 MiB、direct INT8 2,744.1 MiB、
qWD 2,740.3 MiB。由于未在训练前后调用 `torch.cuda.reset/max_memory_allocated`，报告将
GPU 峰值保留为 `null`，不以 `nvidia-smi` 瞬时值替代。

结构化汇总见 `summary.json`，每轮证据见 `raw/4gpu_trial*.json`。
