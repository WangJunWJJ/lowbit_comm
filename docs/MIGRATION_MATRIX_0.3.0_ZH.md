# lowbit_comm 0.3.0 功能与性能迁移矩阵

## 状态定义

- `ORACLE`：旧实现仅作为数值或性能参考；
- `PLANNED`：已进入计划，尚未实现；
- `READY`：新接口、正确性和性能门禁通过；
- `EXPERIMENTAL`：可显式使用，不进入 `auto`；
- `REMOVED`：设计上不迁移。

只有 `READY` 计入 0.3.0 完成度。

## 功能矩阵

| 能力 | 0.2.x 来源 | 0.3.0 目标 | 初始状态 | 必须验证 |
|---|---|---|---|---|
| INT8 quant/dequant | CUDA csrc | CUDA Backend operation | READY | Task 7 codec oracle 与安全 extension load |
| INT4 quant/dequant | CUDA csrc | 显式实验 operation | EXPERIMENTAL | codec 支持；不进入 auto |
| fused quant-pack | CUDA csrc | CUDA lowering | READY | caller-owned payload；Task 7/8/9 真机门禁 |
| gathered dequant-reduce | CUDA csrc | FullTensor/EF lowering | READY | ReducedShard/FullTensor 2/4 卡门禁 |
| qsend/qrecv | communication | typed CUDA P2P executable | READY | logical tag/group/同步语义；A6000 双卡验证 |
| iqsend/iqrecv | communication | typed async CUDA P2P executable | READY | Work 持有 source/metadata/payload 至双句柄完成 |
| dynamic P2P | CPU metadata | 24×int64 device metadata packet | READY | shape/dtype/schema/version/layout generation/flags |
| qall_gather_dyn | collectives | Dynamic Gather executable | READY | 24×int64 metadata；A6000 2/4 卡；无 all_gather_object |
| native collectives | native_collectives | CUDA native facade | READY | 9 类 primitive；A6000 2/4 卡 conformance |
| compressed all-gather | collectives | explicit FullTensor algorithm | READY | A6000 2/4 卡；rank gap=0；不进入 auto |
| ring/tree/p2p | transports | topology lowering | READY | 3/6 rank schedule；P2P 双卡；无固定卡数分支 |
| overlap-* | Work/transport | async lowered schedule | READY | completion chain、event ordering、无 device-wide synchronize |
| compressed reduce-scatter | transport | Quantized ReducedShard | READY | Task 8；无 final all-gather；A6000 2/4 卡 |
| compressed FullTensor | 混合 restore | 两段 QuantizedWire | READY | Task 9；两段量化 collective；A6000 2/4 卡 |
| hierarchical | transport | topology pass | READY | 单机 2/4、双机 4/8 rank 零误差；显式策略 |
| Gradient EF | quantization | DDP Adapter state | READY | Task 10 local reconstruction、事务提交、bucket rebuild |
| Parameter EF/qWD | examples/communication | Sharded Adapter | READY | add-writeback、checkpoint、周期 refresh；A6000 2/4 卡 100-step |
| ReducedShard consumer | optim/examples | Sharded Adapter | READY | FP32 master shard、layout 校验、AdamW reference 对齐 |
| CANN/HCCL | ascend | 后续 Backend | EXPERIMENTAL | 不阻塞 CUDA 0.3.0 |
| ParaScale plugin | plugin.py | 外部 Adapter 集成 | REMOVED | 迁移文档 |
| 旧 Python API | ccdl_comm | 不兼容 | REMOVED | 新 API 示例 |

## 已知性能信号与重测要求

| 场景 | 历史信号 | 0.3.0 要求 |
|---|---|---|
| 2 卡 FullTensor all-gather EF | 曾约 +9.28% vs Native | 新路径相对对应旧快路径不低于 -2% |
| 4 卡 FullTensor all-gather EF | 曾约 -17.71% vs Native | 不进 auto；两段量化路径重新竞速 |
| 2/4 卡 ReducedShard synthetic | 曾有显著加速 | 复现并证明无 full restore |
| 4 卡真实 sharded workload | 曾约 +7.57% vs Native | 三次交替中位数、收敛与 rank 一致 |
| qWD | 曾约 +6.81% vs Native | time-to-quality、refresh 与 checkpoint |

历史数据不自动成为 `auto` 证据。正式结果必须记录 GPU、拓扑、软件版本、world size、
dtype、bucket、batch、wire、output、EF、commit、Kernel ABI、原始样本、中位数与离散度。

## 删除门禁

旧控制面只能在以下条件同时满足后删除：

1. 所有非 EXPERIMENTAL/REMOVED 行达到 READY；
2. 洁净 clone 的 wheel 构建、安装、测试成功；
3. 单机 2/4 卡与双机 4/8 卡正确性通过；
4. 生产快路径性能门禁通过；
5. 新代码对旧 Python 控制面依赖扫描为空；
6. README、示例、CHANGELOG 和迁移文档只展示 0.3.0 API。
