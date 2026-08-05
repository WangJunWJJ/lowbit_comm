# ReducedShard consumer A6000 端到端验证

## 结论

Task 5 的 2 卡与 4 卡门禁均通过。`ccdl_sharded_sgd` 在 44,971,744 参数 FP16 MLP、batch 16/rank 下，相对 native DDP 的端到端吞吐提升分别为 **3.55%** 和 **7.88%**。4 卡相对现有 CCDL full-gradient all-gather 路径提升 **44.13%**；2 卡则低 **2.11%**，但仍高于 native DDP。

| GPU | 模式 | 吞吐中位数（samples/s） | P50（ms） | P95（ms） | 峰值显存（bytes） | 最终 loss |
|---:|---|---:|---:|---:|---:|---:|
| 2 | native DDP | 2142.253 | 14.950 | 15.081 | 421813760 | 7.4645703 |
| 2 | CCDL full gradient | 2266.130 | 14.051 | 14.578 | 967551488 | 7.4645698 |
| 2 | CCDL sharded SGD | 2218.238 | 14.427 | 14.574 | 694814208 | 7.4645693 |
| 4 | native DDP | 3029.345 | 21.069 | 21.411 | 421813760 | 7.4692484 |
| 4 | CCDL full gradient | 2267.383 | 28.155 | 28.709 | 1060072448 | 7.4692487 |
| 4 | CCDL sharded SGD | 3268.049 | 19.622 | 19.928 | 654962176 | 7.4692479 |

每种 GPU 数量和模式均使用三个全新 `torchrun` 进程，表格采用吞吐中位数，并从对应的中位吞吐 run 提取其余指标。18 份原始 JSON 位于 `raw/`，完整数值和代表 run 映射位于 `summary.json`。

## 正确性与执行路径

- Gloo 与 NCCL 两 rank 精确 oracle 均得到 `max_rank_difference=0.0`，workspace 指针稳定。
- 18 个训练 run 均为有限 loss，代表 run 的 rank 参数最大差异为 0。
- sharded run 均确认 `cuda_extension`、`effective_strategy=compressed`、`fast_path=cuda_reduced_shard`、`output_layout=shard`，且没有 fallback。
- 2 卡门禁要求 sharded/native 不低于 0.95，实际为 1.0355；4 卡要求 sharded/full-gradient 严格大于 1.0，实际为 1.4413。

## 五阶段归因

| GPU | backward + flatten | compressed reduce-scatter | local update | parameter all-gather | writeback |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.734 ms | 5.314 ms | 0.204 ms | 6.915 ms | 0.288 ms |
| 4 | 1.925 ms | 7.472 ms | 0.107 ms | 9.809 ms | 0.292 ms |

参数恢复（parameter all-gather + writeback）占平均 step 的 49.93%（2 卡）和 51.58%（4 卡），是当前最大的性能边界；compressed reduce-scatter 分别占 36.84% 和 38.15%。因此后续若仍要求 replicated model，每步完整参数 all-gather 会限制上限；更高收益需要长期保持参数分片、让计算直接消费本地 shard，或把参数预取与下一段计算重叠。

4 卡相对 2 卡不会获得线性 2 倍吞吐：本工作负载将 global batch 从 32 增至 64，但跨 NUMA 的 GPU0-3 拓扑包含 `SYS` 链路，同时 reduce-scatter 和完整参数 all-gather 都随 rank 增加。实测 sharded 4 卡/2 卡吞吐为 1.473×，瓶颈正是上述两段通信，而非 local shard update。

## 环境与边界

- 训练源码：`e713b92714f327906ee7d5fc60859d3c009460ae`；门禁直接入口修复：`523baff`。
- A6000 驱动 550.142；固定镜像 `ccdl-comm-a6000:cu126-torch25`，PyTorch 2.5 nightly、CUDA 12.6、NCCL 2.22.3。
- 2 卡使用 GPU0-1；4 卡使用 GPU0-3，后者跨两个 NUMA 区域。
- 22 个 synthetic step 只能证明短程数值一致性和端到端速度，不能证明真实数据集最终精度、完整收敛步数或泛化能力。
