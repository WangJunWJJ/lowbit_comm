# ReducedShard Consumer 与分片训练设计

日期：2026-08-05

## 1. 背景与目标

当前 CCDL 的 compressed reduce-scatter 已能直接返回 `ReducedShard`，并在
A6000 四卡通信微基准中相对“分片后恢复完整梯度”的路径取得约 2.38 倍延迟
收益。但现有端到端 DDP 示例必须返回完整梯度 bucket，最终 FP16 all-gather
会抵消压缩通信收益。

本阶段在 CCDL 仓库内建立通用 ReducedShard consumer 契约，并在 `examples/`
实现 ZeRO-2 风格的端到端训练示例。每个 rank 只更新自己拥有的参数分片，随后
收集更新后的 FP16 参数分片，使所有 rank 继续持有一致的完整参数。实现不得依赖
ParaScale、PyTorch FSDP 私有接口或 DDP 通信 hook。

## 2. 范围

### 2.1 本阶段包含

- 后端无关的 ReducedShard consumer 协议；
- 确定性的扁平参数与梯度分片布局；
- caller-owned ReducedShard 输出 workspace；
- 无 momentum SGD 的本地参数分片更新；
- 基于连续缓冲区和 `all_gather_into_tensor` 的 FP16 参数恢复；
- `examples/sharded_training.py` 端到端训练示例；
- native DDP、CCDL full-gradient all-gather 和 CCDL sharded consumer 三种模式
  的同口径正确性与性能比较；
- A6000 单机 2 卡和 4 卡验证。

### 2.2 本阶段不包含

- ParaScale 集成；
- PyTorch FSDP 或其他框架的私有 hook；
- AdamW、momentum、混合精度 master weights 和动态参数组；
- 稀疏梯度；
- 参数永久分片或模型计算分片；
- 多机性能门禁。

上述能力只能在本阶段正确性和四卡性能门禁完成后单独设计。

## 3. 方案选择

采用“通用 consumer 契约 + ZeRO-2 风格示例”。不采用只做张量微基准的方案，
因为它不能证明端到端 loss、权重一致性和训练吞吐；也不在本阶段构建完整通用
分片优化器框架，避免把 CCDL 扩张成训练框架并掩盖通信瓶颈。

## 4. 架构与职责

### 4.1 Core consumer 契约

新增后端无关协议，表达“谁消费分片”和“消费完成后返回什么”，协议只依赖
`ReducedShard` 与不可变布局元数据。协议不得 import `torch`，也不得管理进程组。

建议接口：

```python
class ReducedShardConsumer(Protocol):
    def consume(self, reduced: ReducedShard) -> object:
        """Consume one rank-local reduced shard and return a completion value."""
```

`object` 返回值允许 CUDA backend 返回 tensor、event-aware work 或未来扩展的
consumer result，同时保持 Core 后端无关。

### 4.2 分片布局

新增不可变 `FlatShardLayout`，至少记录：

- `original_numel`；
- `padded_numel`；
- `shard_numel`；
- `world_size`；
- 当前 rank 的 `shard_index`、`shard_offset`、`valid_numel`；
- 参数的扁平 offset、shape、dtype 与 requires-grad 顺序签名。

布局必须与 `ReducedShard` 的逻辑范围完全一致。参数遍历顺序固定为
`model.parameters()` 顺序，梯度缺失时写入零值，不允许跳过参数从而改变 offset。

### 4.3 Torch 示例 consumer

Torch 相关实现属于 `examples/training/`，不进入 Core：

- 将模型参数预先映射到连续 flat parameter buffer；
- backward 后将梯度写入连续 flat gradient buffer；
- 调用已编译的 compressed reduce-scatter plan，输出到 caller-owned shard buffer；
- 本地执行 `parameter_shard -= learning_rate * gradient_shard`；
- 使用预分配连续 full parameter buffer 执行
  `torch.distributed.all_gather_into_tensor`；
- 仅把有效范围写回原模型参数，padding 永远不进入模型。

第一版使用无 momentum SGD，不维护额外 optimizer state。示例 consumer 的接口和
实现保持独立，以便后续增加 AdamW consumer，而不修改通信层。

## 5. 数据流与有序完成

单步训练数据流为：

```text
forward
  -> backward
  -> flatten gradients into reusable full-gradient workspace
  -> compressed reduce-scatter into caller-owned ReducedShard workspace
  -> wait/stream-order the ReducedShard before local SGD update
  -> update local FP16 parameter shard
  -> all_gather_into_tensor into reusable full-parameter workspace
  -> write gathered parameters back to model views
  -> next forward
```

通信、consumer 更新和参数恢复必须位于明确的 CUDA stream 顺序中。默认实现允许
当前 stream 通过 event 等待通信完成，不允许通过 CPU 轮询制造隐式
`cudaDeviceSynchronize`。首轮可以在 step 边界同步以获得可靠计时，但同步必须被
计入端到端延迟。

caller-owned ReducedShard buffer 的生命周期覆盖通信完成和本地 SGD 更新；只有
consumer 更新完成事件可证明结束后才允许 workspace pool 复用。

## 6. 正确性约束

每个同口径运行必须满足：

1. 模型结构、初始化 seed、数据顺序、global batch、学习率、精度和训练步数一致；
2. 每步参数恢复后，各 rank 完整参数逐元素一致；
3. loss 有限并在测量区间下降；
4. sharded consumer 的最终 loss 相对 native DDP 偏差不超过批准阈值；
5. ReducedShard metadata 与布局的 rank、world size、offset、valid/padding numel 一致；
6. 不允许静默 fallback，执行结果必须报告实际策略和 CUDA extension capability；
7. 任何通信、布局或 workspace ownership 错误必须终止运行并保存失败报告。

INT8 的最终 loss 阈值沿用端到端门禁的相对差异上限 `0.02`。通信张量的相对 L2
继续使用 `0.02`，但该数值只用于短程数值正确性，不代表最终任务精度。

## 7. 性能与测量

### 7.1 对照模式

- `native_ddp`：PyTorch FP16 DDP；
- `ccdl_full_gradient`：当前 CCDL INT8 all-gather/full-gradient 路径；
- `ccdl_sharded_sgd`：compressed reduce-scatter、分片 SGD、FP16 参数 all-gather。

三种模式采用独立进程运行，固定 GPU、seed、模型、数据与 warm-up。至少执行三轮，
以中位数作为性能结论，保留全部原始 JSON。

### 7.2 指标

- samples/s；
- step P50/P95；
- peak allocated bytes；
- backward/gradient flatten 时间；
- compressed reduce-scatter 时间；
- local shard update 时间；
- FP16 parameter all-gather 时间；
- parameter writeback 时间；
- loss 与 rank 参数一致性。

### 7.3 验收门槛

- 2 卡：`ccdl_sharded_sgd` 相对 native DDP 不得出现超过 5% 的稳定回退；
- 4 卡：`ccdl_sharded_sgd` 吞吐必须高于当前 `ccdl_full_gradient`；
- 正确性约束必须先通过，才允许报告性能收益；
- 若四卡未通过，报告必须把主要耗时归因到 compressed reduce-scatter、参数
  all-gather、flatten/writeback 或同步等待中的具体阶段，不得仅报告总吞吐。

首轮不强制 `ccdl_sharded_sgd` 超过 native DDP；该目标取决于参数 all-gather、模型
计算通信比和 A6000 拓扑。只有测量结果满足时才能宣称相对 native 加速。

## 8. 测试策略

### 8.1 本地单元测试

- `FlatShardLayout` 的整除、padding、空 shard 与非法 metadata；
- consumer 协议不依赖 torch；
- flat parameter/gradient offset 和写回正确；
- 无梯度参数补零且不改变布局；
- caller-owned output 指针稳定；
- 参数 all-gather 使用连续预分配输出；
- 失败时 workspace 不提前复用。

### 8.2 分布式动态测试

- CPU/Gloo 小模型 2 rank 正确性 smoke；
- A6000 2/4 rank CUDA extension 与 metadata smoke；
- 每步 rank 参数一致性；
- native DDP 与 sharded SGD 的短程 loss 对照；
- 2/4 卡分阶段计时和端到端性能门禁。

## 9. 交付物与提交边界

实施按独立能力提交：

1. Core consumer 与布局契约；
2. Torch flat parameter/gradient 工具和 SGD consumer；
3. `examples/sharded_training.py` 与正确性 smoke；
4. A6000 2/4 卡性能测量、门禁与报告；
5. 若门禁暴露单一热点，再以独立 `perf` commit 优化该热点。

每个提交必须先完成对应 TDD 红—绿循环，并遵循
`<type>(<scope>): <subject>` 提交格式。
