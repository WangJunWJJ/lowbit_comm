# 四卡 ReducedShard 参数压缩恢复流水线设计

## 1. 背景与目标

CCDL 当前已经能够通过 compressed reduce-scatter 返回 `ReducedShard`，并由
`ShardedSGDConsumer` 只更新本 rank 对应的参数和优化器状态。A6000 四卡实测中，
本地 shard update 只占平均 step 的约 0.5%，compressed reduce-scatter 约占 38%，
而更新后的完整参数 all-gather 与 writeback 约占 51%。继续优化梯度端的 fused
kernel 已不能解决主要瓶颈。

本设计在 CCDL 独立仓库内增加通用的分片优化器消费与压缩参数恢复流水线：

1. 直接消费 `ReducedShard`，不恢复完整梯度；
2. 只在本 rank 更新参数 shard 和 optimizer state；
3. 将更新后的参数 shard 量化为 INT8；
4. 以 INT8 执行参数 all-gather；
5. 用单个 CUDA kernel 将所有 rank payload 反量化并直接写入完整参数 flat view；
6. 通过 bucket、CUDA stream 和 event 尽可能重叠通信与本地更新。

近期性能门槛是四卡端到端吞吐不低于现有 fused CCDL，并优先争取相对 Native
DDP 获得至少 5% 的稳定中位数提升。性能门槛不是正确性降级的理由。

## 2. 方案选择

### 2.1 采用方案

采用 `ReducedShard + INT8 参数 all-gather + fused gathered-dequant/writeback`。
它保留普通 replicated model 的前向计算接口，同时将不可避免的参数恢复通信从
FP16/FP32 压缩到 INT8，并消除逐 rank 反量化和中间完整参数张量复制。

### 2.2 未采用方案

- 仅保留 FP 参数 all-gather：兼容性最好，但约一半 step 时间仍不可优化。
- 永久参数分片并 layerwise prefetch/reshard：理论上限最高，但需要介入 module
  execution、autograd 和层级调度，已经接近 FSDP 型训练框架。本阶段只保留接口
  扩展空间，不在 CCDL core 中实现模型执行引擎。
- 为四卡建立专用分支或专用 kernel：会破坏任意 world size 的通用性。实现必须
  使用运行时 `world_size`，只将当前验证范围限制为 2--8 rank。

## 3. 架构与职责

### 3.1 `ShardedOptimizerConsumer`

新增生产级、后端无关的 consumer contract 实现，职责包括：

- 接收一个与已注册参数布局匹配的 `ReducedShard`；
- 依据 `logical_range` 和 `valid_numel` 选择本 rank 的有效参数区间；
- 只维护并更新本 rank 的 optimizer state；
- 返回包含更新后参数 shard、布局版本和 completion dependency 的结果；
- 拒绝 world size、rank、layout version、dtype 或 numel 不匹配的输入。

第一阶段实现 SGD 语义作为可验证参考，不把 AdamW 算法硬编码进通信层。参数更新
算法通过小型 `ShardUpdateRule` protocol 注入，以便后续增加 AdamW 而不修改
transport。

### 3.2 `CompressedParameterRestore`

负责将更新后的本地参数 shard 恢复为所有 rank 一致的完整参数：

- fused quant-pack 本地参数 shard；
- 分配或租用固定 stride 的 INT8 send/gather workspace；
- 调用 `all_gather_into_tensor` 聚合所有 rank payload；
- 调用 gathered-dequant/writeback kernel；
- 返回可等待的 completion object；
- capability 不满足时回退到 FP16 参数 all-gather。

该组件只负责通信和布局恢复，不执行 optimizer 数学。

### 3.3 `ShardedStepPipeline`

按参数 bucket 调度 consumer 与 restore：

- 每个 bucket 依次经过 reduce-scatter、local update、quant-pack、INT8 gather 和
  writeback；
- communication stream 上执行 collective，restore stream 上执行可安全并行的
  kernel；
- event 表示 gathered payload 可读和参数 bucket 可用于下一次计算；
- 限制最大在途 bucket 数，超过预算时等待最早 bucket，避免无界 workspace；
- step completion 只有在所有参数 bucket writeback 完成后才可见。

第一阶段不承诺与同一 step 的 backward 计算重叠，因为示例训练在 backward 完成后
才获得完整梯度布局。接口保留 `consume_bucket()`，后续可由 autograd bucket producer
提前触发。

## 4. 数据流

```text
backward gradients
    -> bucket flat view
    -> INT8 compressed reduce-scatter
    -> fused dequant + reduce + mean
    -> ReducedShard
    -> local optimizer shard update
    -> fused INT8 quant-pack
    -> async INT8 all_gather_into_tensor
    -> single-kernel gathered-dequant + writeback
    -> full parameter bucket ready event
```

多个 bucket 允许以下流水：

```text
bucket N:   reduce-scatter -> update -> quant -> gather -> restore
bucket N+1:                  reduce-scatter -> update -> quant -> gather -> restore
```

具体重叠程度由 NCCL stream 顺序、bucket 大小与 workspace 预算决定，不在接口层伪造
异步完成。

## 5. 单 kernel gathered-dequant + writeback

INT8 all-gather 输出为一块 rank-strided buffer：

```text
[rank0 payload | stride padding | rank1 payload | ... | rankN payload]
```

kernel 对每个逻辑输出元素执行：

1. 从全局输出下标推导来源 rank、rank 内元素下标和 quantization group；
2. 从相应 payload 读取 INT8 值和该 group 的 scale；
3. 按目标 dtype 反量化；
4. 直接写入目标参数 flat view 的最终位置。

kernel 不构造 rank 级临时反量化 tensor，不执行 `torch.cat`，也不需要随后把完整
restored workspace 再 `copy_` 到参数。目标参数连续且 layout 完全匹配时使用 direct
writeback；否则写入池化 restored workspace，再由受控 scatter fallback 写入原参数。

kernel 必须支持最后一个不完整 group，只读取带 padding 的合法 payload，并只写
`original_numel` 范围内的元素。world size、payload stride 和 dtype 都是运行时参数，
但 fast path 仍 capability-gated 为 linear INT8、group size 64、top-k 0、non-compact。

## 6. 数值与一致性语义

- 所有 rank 对同一组更新后参数 shard 执行 collective，接收完全相同的 gathered
  payload，因此恢复后的完整参数必须逐元素一致。
- 参数量化引入一次可测的 INT8 舍入误差。第一阶段不对参数使用 error feedback，
  避免 residual 成为隐藏的参数状态并改变 optimizer 语义。
- 提供 `restore_dtype=fp16` fallback，用于精度门禁或不满足 fused capability 的布局。
- 每个 step 完成前必须等待全部 writeback event，下一次 forward 不得读取仍在恢复的
  参数 bucket。
- padding 元素不能进入 optimizer update，也不能写出模型逻辑参数范围。

## 7. Workspace 与生命周期

workspace key 至少包含：bucket identity、logical/padded numel、dtype、world size、bit、
group size、device 和 stream ownership。池中分别复用：

- quantized send payload；
- gathered INT8 payload；
- 非 direct-writeback 场景的 restored output；
- optimizer shard state。

每个 lease 由 CUDA event 保护。仍被 collective 或 kernel 使用的 buffer 不得重新发放。
异常路径必须释放尚未提交的 lease，并让已提交 CUDA work 保持资源存活到完成。

## 8. 错误处理与 fallback

- 元数据或参数布局不匹配：抛出明确异常，不进行 silent fallback。
- CUDA extension、fused symbol 或 `all_gather_into_tensor` 不可用：回退到已验证的
  FP16 parameter gather。
- fast-path policy 不支持：记录 capability reason 并回退，不把运行时正常拒绝当成
  CUDA 错误。
- collective 已启动后失败：completion object 传播原始异常，禁止返回部分恢复参数。
- workspace 预算不足：等待最早在途 bucket；不临时进行无界分配。

## 9. 公共接口草案

```python
class ShardUpdateRule(Protocol):
    def update(self, parameter_shard, gradient_shard, state, *, step: int): ...


class ShardedOptimizerConsumer:
    def consume(self, reduced: ReducedShard) -> UpdatedParameterShard: ...


class CompressedParameterRestore:
    def restore(self, updated: UpdatedParameterShard, *, out=None, async_op=True): ...


class ShardedStepPipeline:
    def consume_bucket(self, reduced: ReducedShard, *, parameter_view): ...
    def finish_step(self): ...
```

具体返回值使用 CCDL `Work`/completion 语义，不暴露裸 NCCL handle。

## 10. 示例与测试

新增 `examples/training/compressed_sharded_optimizer.py`，提供三种同口径模式：

- Native DDP；
- 现有 ReducedShard + FP parameter gather；
- 新 ReducedShard + compressed parameter restore pipeline。

测试按 TDD 分层进行：

1. core 单测：layout、padding、update rule、版本校验和异常回滚；
2. transport 单测：workspace ownership、fallback、async completion；
3. CUDA 测试：FP16/BF16/FP32、2/4/8 rank-strided payload、非整组尾部；
4. 分布式正确性：2/4 卡参数最大 rank 差异为 0，与未融合参考误差在预设门槛内；
5. 微基准：量化、INT8 gather、单 kernel restore 各阶段计时；
6. 端到端：四卡 A6000 三模式交替至少 3 次，丢弃统一 warmup 后取吞吐中位数；
7. 真实训练：21 GB 数据集完整 epoch、validation、checkpoint，无 NaN/Inf/CUDA/NCCL
   错误。

性能结果必须同时报告 Native DDP、旧 sharded consumer、新 pipeline 的原始三次吞吐、
中位数、P50/P95 step、通信阶段占比和 loss。若新路径未通过性能门槛，保持显式 opt-in，
不得进入默认策略。

## 11. 非目标

- 不在本阶段实现通用 FSDP、ZeRO 或 layerwise sharded execution engine；
- 不修改 ParaScale；
- 不重写 NCCL；
- 不为四卡硬编码拓扑或 world size；
- 不默认启用尚未通过真实训练精度门禁的参数 INT8 恢复。
