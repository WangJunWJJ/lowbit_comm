# PSI Policy FP16 严格交替 A/B 测试

## 结论

在同一台 A6000 服务器、同一 Docker 镜像、同一 GPU 集合和同一最终 CUDA
扩展上，Native DDP、未融合 CCDL 和 fused CCDL 分别交替运行 3 次完整
epoch。所有运行均明确记录 `Mixed Precision: fp16`，统计丢弃前 20 个训练
step 后，以全局 batch 除以平均 step time 计算单次吞吐，再取 3 次吞吐中位数。

| GPU | 模式 | 3 次吞吐（samples/s） | 中位数 | vs Native | vs 未融合 |
|---:|---|---|---:|---:|---:|
| 2 | Native DDP | 197.376 / 200.580 / 195.956 | **197.376** | 基线 | -0.76% |
| 2 | CCDL 未融合 | 203.579 / 191.093 / 198.896 | **198.896** | +0.77% | 基线 |
| 2 | CCDL fused | 193.961 / 205.360 / 204.788 | **204.788** | **+3.76%** | **+2.96%** |
| 4 | Native DDP | 358.885 / 363.490 / 356.509 | **358.885** | 基线 | -3.12% |
| 4 | CCDL 未融合 | 370.450 / 371.098 / 368.306 | **370.450** | **+3.22%** | 基线 |
| 4 | CCDL fused | 371.030 / 366.033 / 374.789 | **371.030** | **+3.38%** | +0.16% |

因此，本轮严格 FP16 端到端口径下，fused CCDL 相对 Native DDP 的训练吞吐
提升为双卡 **3.76%**、四卡 **3.38%**；相对未融合 CCDL 的额外收益为双卡
**2.96%**、四卡 **0.16%**。四卡 fused 与未融合的差距小于本轮运行波动，不能
据此声称融合在四卡端到端场景取得显著额外加速。

## 时延与扩展效率

| GPU | 模式 | 平均 step 中位数 | P95 step 中位数 | 4/2 卡吞吐比 |
|---:|---|---:|---:|---:|
| 2 | Native DDP | 162.127 ms | 175.022 ms | - |
| 2 | CCDL 未融合 | 160.888 ms | 188.877 ms | - |
| 2 | CCDL fused | **156.259 ms** | **166.064 ms** | - |
| 4 | Native DDP | 178.330 ms | 199.854 ms | 1.818x |
| 4 | CCDL 未融合 | 172.763 ms | 212.861 ms | 1.863x |
| 4 | CCDL fused | **172.493 ms** | **196.634 ms** | 1.812x |

四卡全局 batch 是双卡的 2 倍，但吞吐只达到约 1.81--1.86 倍，说明剩余瓶颈
包括单机四卡通信拓扑、量化/反量化开销以及 DDP 完整 bucket 恢复；本结果不支持
“四卡必然是双卡 2 倍吞吐”的假设。

## 收敛与数值结果

| GPU | 模式 | epoch train loss | val loss |
|---:|---|---:|---:|
| 2 | Native DDP | 3.937248 | 2.044303 |
| 2 | CCDL 未融合 / fused | 3.881366 | 2.047609 |
| 4 | Native DDP | 4.184546 | 2.982103 |
| 4 | CCDL 未融合 / fused | 4.187568 | 2.945411 |

同一 GPU 数下，未融合和 fused 的 loss 逐步记录及最终验证 loss 完全一致，证明
融合只改变执行方式，没有改变当前 INT8 通信语义。双卡 CCDL 的 val loss 相对
Native 高 0.003307（约 0.16%），四卡则低 0.036692；方向不一致，且只训练一个
epoch，因此只能证明没有发散、NaN 或额外收敛步数，不能证明最终任务精度等价。

## 测试口径

- 数据集：`pis-policy-v1-align-10015/open-paper-bag_C8JXLG`，
  21,586,646,073 bytes（约 21 GB）。
- 模型：三视角 diffusion policy，约 44,956,124 参数；DiT 子模块 8,909,212
  参数；AdamW；1 epoch；完整 validation。
- 每 rank batch size 16；双卡全局 batch 32、1404 step；四卡全局 batch 64、
  702 step。
- 双卡固定物理 GPU 1、2；四卡固定物理 GPU 1、2、3、4。
- 镜像：`ccdl-pis-policy:a6000-torch25`，镜像 ID
  `sha256:89f988d94bcff81806d95b27522a632b258a578b52587635cb255404ed386335`。
- 最终 CUDA 扩展 SHA-256：
  `844946c5aa126ad072850e41626721b66de54b8985b0642f0fb8747f4f0f4799`。
- 轮换顺序：trial1 `Native -> 未融合 -> fused`；trial2
  `未融合 -> fused -> Native`；trial3 `fused -> Native -> 未融合`。
- 每轮都生成结构化日志、540,105,768-byte checkpoint 和完成标记；18/18
  轮均确认 FP16，未发现 traceback、CUDA error 或 NCCL warning。

## 测试期间发现并闭环的问题

真实模型的 DDP bucket 分片为 22,478,062 个元素，不能被 group size 64 整除。
原 fused gathered-dequant kernel 虽然已收到带尾组 padding 的合法 payload，却错误
要求 `shard_numel % 64 == 0`，导致融合路径拒绝。修复采用向上取整的 group 数，
只写逻辑 `shard_numel` 个输出元素。

该问题按 TDD 闭环：新增 FP16/BF16/FP32 非整组尾分片用例，修复前 3 个用例均
失败，修复后整个 fused restore CUDA 测试文件 **30 passed**。为了保持严格同版本
口径，修复前的 trial1 已隔离，双卡和四卡 trial1 三种模式全部使用最终扩展重跑。

完整结构化结果见 [summary.json](summary.json)。
