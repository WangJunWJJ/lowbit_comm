# qWD 分片参数通信设计

## 1. 背景与目标

当前 CCDL 分片训练路径已经能够执行 compressed reduce-scatter、rank-local
AdamW 更新和 INT8 参数 all-gather，但参数恢复直接量化更新后的参数 shard。四卡
A6000 真实训练中，该路径虽然相对 Native DDP 提升 7.57% 吞吐，却使 validation
loss 从 2.938473 上升至 4.704630。各 rank 参数逐元素一致，因此问题来自参数量化
误差持续进入下一次 forward，而不是 rank 分歧。

本设计将参数通信核心改为 SDP4Bit 的 quantized weight difference（qWD）：每个
rank 维护 FP32 master shard，并量化通信 master shard 与当前 replicated model copy
对应 shard 的差值。未被本次量化准确传递的差值自然保留到下一步，形成独立于梯度
压缩的 Parameter Error Feedback（Parameter EF）。

目标如下：

1. AdamW 数学和优化器状态以 FP32 rank-local master shard 为唯一权威状态；
2. forward/backward 使用 replicated FP16/BF16 model copy；
3. 默认参数通信使用 INT8 qWD，而不是直接量化参数；
4. 支持 warmup、周期 FP16 refresh 和误差触发 refresh；
5. mixed-bit 由明确策略和运行时指标决定，不依赖参数名称猜测；
6. 保持任意 world size 的接口语义，不为 2/4/8 rank 硬编码实现；
7. 在恢复可接受精度的同时，保持或提高当前四卡 compressed 路径吞吐。

参考实现和算法依据包括 [SDP4Bit](https://www.proceedings.com/content/079/079017-0279open.pdf)、
[QSDP](https://openreview.net/pdf?id=Nqp8A5IDzq)、
[LoCo](https://arxiv.org/pdf/2407.04480) 和旧 CCDL 的 4/8-bit、TopK、随机舍入、
group quantization 与异步 collective 实现。

## 2. 方案比较与选择

### 2.1 采用方案：FP32 master + qWD + 安全 refresh

每个 rank 仅更新自己的 FP32 master shard。参数通信计算：

```text
delta_local = master_shard_fp32 - model_copy_local_fp32
delta_hat   = dequantize(all_gather(quantize(delta_local)))
model_copy += delta_hat
```

下一步重新计算 `master - model_copy`。上一步未恢复的量化残差会自动包含在新的
差值中，不需要另行修改 AdamW 梯度或 moment。该方案直接复用当前 compressed
parameter restore 的 INT8 collective 和 gathered-dequant/writeback 能力，同时避免
把有偏量化误差永久写入 FP32 master。

### 2.2 未采用方案：直接量化参数 + 显式 residual

该方案需要定义 residual 加在参数、更新量还是梯度上，并处理 residual 与 AdamW
master、checkpoint 和 refresh 的一致性。直接量化参数还会把量化后的参数作为下一步
优化起点，收敛假设更强。它不作为默认参数路径。

### 2.3 未采用方案：随机无偏权重量化

QSDP 通过随机量化满足无偏条件，理论基础充分，但需要新的随机数状态、确定性复现
规则和 kernel。CCDL 先采用与现有 deterministic INT8 kernel 兼容的 qWD；随机舍入
保留为 `QuantizationPolicy` 的可选能力。

### 2.4 未采用方案：默认 INT4 + LoCo EMA

INT4 可进一步减少带宽，但通常需要 Hadamard/异常值隔离等平滑手段。LoCo EMA
主要针对梯度误差。两者均保留为后续 capability-gated 实验策略，不与第一版 qWD
正确性改造捆绑。

## 3. 状态模型与所有权

### 3.1 `MasterParameterShard`

每个 rank 持有一个连续 FP32 tensor，长度为 `layout.shard_numel`，其中仅
`valid_numel` 范围参与 optimizer update。它是 AdamW 更新、checkpoint 和恢复的
权威参数状态。padding 必须保持为零，不得进入 norm、weight decay 或量化误差统计。

AdamW 的 `exp_avg`、`exp_avg_sq` 同样保持 FP32 且仅 rank-local。checkpoint 必须保存：

- layout identity/version；
- optimizer step；
- FP32 master shard；
- FP32 moments；
- qWD policy/scheduler state。

### 3.2 `ReplicatedModelCopy`

forward/backward 参数是完整 replicated flat storage，dtype 为 FP16 或 BF16。它不是
optimizer 权威状态。每次 qWD restore 或 FP refresh 完成后，所有 rank 的 model copy
必须逐元素一致，下一次 forward 才能开始。

model copy 对应的 local shard 只是完整 flat storage 的 view，不单独分配持久参数副本。
计算 qWD 时把该 view 转换为 FP32，与 master shard 相减。后续 fused kernel 可以直接
读取低精度 model copy 并以 FP32 accumulator 计算差值，消除临时转换 tensor。

### 3.3 Parameter EF

Parameter EF 定义为隐式状态：

```text
parameter_error_t = master_t - model_copy_t
```

它不以独立 residual tensor 重复存储。qWD 完成后，`master - model_copy` 自动等于尚未
被准确传递的误差。Parameter EF 不修改 gradient、AdamW moments 或 learning rate。

若未来增加 LoCo-style EMA，必须作为单独策略和 checkpoint 字段，不能改变默认 qWD
的状态定义。

## 4. 组件边界

### 4.1 `TorchShardedAdamWStep`

负责训练级编排：

1. flatten gradients；
2. compressed reduce-scatter；
3. FP32 转换和全局 norm/clip；
4. 更新 FP32 master shard 与 FP32 moments；
5. 根据 scheduler 选择 qWD 或 FP refresh；
6. 等待参数恢复 completion 后开放下一次 forward。

它不实现 quantization kernel 或 distributed collective。

### 4.2 `ParameterDeltaProvider`

提供后端无关的 qWD 输入 contract：

```python
class ParameterDeltaProvider(Protocol):
    def prepare_delta(
        self,
        master_shard: Any,
        model_shard: Any,
        *,
        out: Any,
        valid_numel: int,
    ) -> Any: ...
```

输入必须同设备、连续且 shard 大小一致；`out` 必须是 FP32 caller-owned workspace。
实现写入 `master.float() - model.float()`，padding 写零，并返回同一个 `out`。

### 4.3 `QuantizedParameterDeltaRestore`

该组件只负责 qWD 通信：

- 接收已准备好的 local FP32 delta shard；
- 根据 `ParameterCommunicationDecision` 选择 INT8 qWD 或 FP refresh；
- INT8 路径复用 quantized send/gather workspace；
- all-gather 后反量化 delta，并累加写入 model copy，而不是覆盖参数；
- FP refresh 路径 gather FP32/FP16 master shards 并覆盖 model copy；
- 返回 CCDL completion object，持有全部 workspace lease 直至 CUDA work 完成。

现有 `TorchCompressedParameterRestore` 保留为兼容的直接参数恢复组件，但新训练示例
不得把它误用为 qWD。qWD 使用独立类名和输入类型，避免单位混淆。

### 4.4 `ParameterCommunicationPolicy`

纯决策组件，不执行 tensor 运算或 collective：

```python
@dataclass(frozen=True, slots=True)
class ParameterCommunicationDecision:
    mode: Literal["fp_refresh", "qwd"]
    bit: Literal[8]
    reason: str


class ParameterCommunicationPolicy(Protocol):
    def decide(
        self,
        *,
        step: int,
        tensor_role: str,
        numel: int,
        relative_error: float | None,
        capability: Any,
    ) -> ParameterCommunicationDecision: ...
```

默认第一版只产生 FP refresh 或 INT8 qWD。接口预留 4-bit，但不允许策略在后端没有
对应 kernel/correctness capability 时返回 4-bit。

管理层开销仅发生在每 bucket 一次的 CPU 纯函数调用，不进入元素级热路径；相同
bucket 的稳定决策可缓存。策略选择发生在 collective 启动前，不引入 CUDA synchronize。

## 5. 默认策略

默认 `SafeInt8QWDPolicy` 使用以下确定规则：

1. 前 `warmup_steps=100` 步使用 FP refresh；
2. 之后默认使用 INT8 qWD；
3. 每 `refresh_interval=512` 步执行一次 FP refresh；
4. 上一步采集到的 `relative_error > 1e-2` 时，下一步执行 FP refresh；
5. norm、bias 等调用方标记为 `sensitive` 的 tensor role 使用 FP refresh；
6. qWD kernel/collective capability 不满足时使用 FP refresh，并记录明确 reason；
7. NaN/Inf、layout/dtype/world-size 不匹配属于错误，禁止 silent fallback。

`relative_error` 定义为：

```text
||delta - delta_hat||_2 / max(||delta||_2, 1e-12)
```

误差统计默认每 `error_check_interval=128` 步采样一次，且由 CUDA 设备侧 reduction
产生。第一版允许 completion 边界读取一个标量；不得在每一步引入全 tensor CPU
同步。若误差测量能力不可用，周期 refresh 仍然生效。

上述数值是安全默认值，不是写死在 kernel 中的协议常量。训练示例和公共构造器必须
显式暴露配置，并把实际生效值写入 benchmark 报告和 checkpoint metadata。

## 6. 数据流与完成语义

```text
backward gradients
    -> compressed reduce-scatter
    -> ReducedShard
    -> FP32 AdamW master-shard update
    -> policy decision
       -> qWD:
          master_fp32 - model_copy_local -> delta workspace
          -> quant-pack INT8
          -> async INT8 all-gather
          -> gathered-dequant-add writeback
       -> FP refresh:
          master shard cast/pack
          -> async FP all-gather
          -> overwrite model copy
    -> parameter-ready CUDA event
    -> next forward
```

qWD 的 gathered kernel 必须执行 add writeback。覆盖式 writeback 会丢失 model copy
基线，属于协议错误。FP refresh 必须执行 overwrite，并使隐式 Parameter EF 归零到
一次 dtype cast 的舍入误差。

CUDA stream 顺序为：optimizer update event -> delta/pack stream -> collective stream
-> restore stream -> parameter-ready event。`CollectiveWork.wait()` 只在调用方需要 CPU
可见完成时同步；正常训练依赖 stream/event，不在 Python callback 中调用全局
`torch.cuda.synchronize()`。

## 7. Workspace 与性能边界

每个 bucket 的 workspace pool 复用：

- FP32 master shard（持久状态）；
- FP32 qWD delta；
- quantized send payload；
- gathered payload；
- FP refresh send workspace；
- 可选误差 reduction scratch。

key 至少包含 device、bucket/layout identity、shard numel、world size、model dtype、
bit、group size、quant type 和 stream ownership。正在被 CUDA event 保护的 lease 不得
复用。steady-state 训练不得产生与参数 numel 成比例的新 allocation。

第一阶段允许独立 delta kernel 和现有 quant-pack kernel，以先验证算法；性能阶段再
融合 `master-model -> quant-pack`，以及 `gathered-dequant -> add-writeback -> error
metric`。融合前后必须保持相同 qWD contract 和测试向量。

## 8. 错误处理和 fallback

- layout、rank、world size、dtype、device、numel 不匹配：立即抛出异常；
- qWD capability 缺失：在 collective 启动前选择 FP refresh；
- INT8 collective 已启动后 kernel 拒绝 payload：传播异常，不允许用另一个 collective
  重试，以免 rank collective 序列不一致；
- workspace in-flight：等待受控 lease 或按预算获取另一个 slot，不进行无界分配；
- master/model/delta 含 NaN 或 Inf：抛出 `FloatingPointError`，不得量化；
- 某 rank 策略决策不一致：用固定长度 metadata packet 校验 mode/bit/layout epoch，
  在 collective 前失败；
- checkpoint policy 与运行配置不同：默认拒绝恢复；显式 `reset_parameter_ef=True`
  时执行一次 FP refresh 后采用新策略。

## 9. Checkpoint 与恢复

checkpoint 不保存 replicated model copy 作为权威模型。保存 FP32 master shards 和
optimizer state，加载后执行以下顺序：

1. 校验 layout 和 world-size reshard contract；
2. 加载/reshard FP32 master 与 moments；
3. 恢复 optimizer step 和 policy scheduler；
4. 强制执行一次 FP refresh 构造一致的 replicated model copy；
5. 清空上一进程生命周期的 workspace/event 状态；
6. 下一步重新开始 qWD 差值闭环。

这样不需要保存独立 parameter residual，也不会从低精度 model copy 恢复 FP32 master。

## 10. 可验证门禁

### 10.1 单元与 CUDA 正确性

- qWD delta 等于 FP32 master 减 FP16/BF16 model copy 的 FP32 值；
- padding 恒为零且不改变 model copy 逻辑区间之外的数据；
- 反量化执行 add writeback，FP refresh 执行 overwrite；
- 一次有损 qWD 后，下一步 delta 精确包含上一步未恢复误差；
- FP refresh 后隐式 Parameter EF 仅剩目标 dtype cast 误差；
- 2/4/8 rank payload 及非整 group 尾部正确；
- workspace pointer 在 steady state 保持稳定；
- capability fallback、collective 后失败和 checkpoint 不匹配均具有确定错误语义。

### 10.2 分布式一致性

- 每步参数恢复后，所有 rank model copy 最大差异为 0；
- FP32 master shard 拼接结果与无压缩 sharded AdamW reference 在规定误差内一致；
- 相同 seed、batch 顺序和 mixed-precision 配置下可重复训练；
- 无 NaN/Inf、NCCL sequence mismatch、use-after-free 或额外全局 CUDA synchronize。

### 10.3 端到端精度

使用现有 21 GB 数据集和 44.956M 参数模型，与 `sharded_fp` 使用完全相同 GPU、
容器、batch、seed、epoch 和 mixed precision：

- 三个完整 epoch 后 validation loss 相对 `sharded_fp` 增幅不超过 2%；
- loss 曲线不得出现持续性平台或发散；
- 若未通过，INT8 qWD 保持 opt-in，且报告 refresh/error metric 后再调整策略。

长期门禁增加固定训练预算下的最终任务指标；三 epoch loss 只作为当前回归门禁，不宣称
等价于完整收敛证明。

### 10.4 性能

在单机四卡 A6000（GPU 1/2/3/4）上交替运行 Native DDP、`sharded_fp`、现有直接参数
INT8 路径和 qWD 路径，每种至少三次并取中位数：

- qWD steady-state 吞吐不得低于现有直接参数 INT8 路径的 98%；
- qWD 吞吐相对 Native DDP 目标提升至少 5%；
- 报告 samples/s、P50/P95 step、reduce-scatter、optimizer、delta/quant、all-gather、
  restore 各阶段占比、显存峰值和 steady-state allocation；
- warmup 和 FP refresh step 单独报告，不混入 qWD steady-state kernel 微基准；
- 未通过性能门禁时不得以降低 refresh 或放宽精度门禁掩盖问题。

## 11. 分阶段交付

1. FP32 master shard、checkpoint 和无压缩 FP refresh reference；
2. qWD delta contract 与隐式 Parameter EF 正确性；
3. INT8 qWD restore、capability fallback 和 CUDA completion；
4. safe policy、warmup、周期/误差触发 refresh；
5. 单机 2/4 卡一致性与精度验证；
6. fused qWD quant-pack、gathered-dequant-add 与 workspace 优化；
7. 21 GB 数据集四卡三次交替端到端性能与精度验收；
8. 通过 INT8 门禁后，再独立设计 INT4、TopK/Hadamard 或 LoCo EMA 实验策略。

每个阶段采用测试先行并独立提交。性能优化不得改变已经通过的 qWD 数学 contract。

## 12. 非目标

- 不修改 ParaScale；
- 不实现模型执行引擎、通用 FSDP 或 ZeRO；
- 不重写 NCCL；
- 不把 INT4、TopK、Hadamard、LoCo EMA 纳入第一版默认路径；
- 不为特定 rank 数硬编码 kernel；
- 不通过放松精度门禁换取表面吞吐；
- 不要求兼容旧 CCDL Python API。
