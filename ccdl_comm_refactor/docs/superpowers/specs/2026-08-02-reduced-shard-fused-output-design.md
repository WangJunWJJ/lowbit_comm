# Task 12.1 ReducedShard 融合归约与输出所有权设计

## 目标

在 Task 12 已验证的 compressed reduce-scatter 数据流上，将多 rank 压缩
payload 的 `dequant -> reduce -> mean -> shard output` 合并为一次 CUDA
kernel，并为最终 ReducedShard 提供可证明安全的零分配输出复用机制。

Task 12.1 完成后，Task 13 的 ring/tree/P2P pipeline 可以直接复用同一个
融合 kernel、输出契约和 stream-safe workspace 生命周期，不再重复实现本地
归约与 buffer ownership。

## 当前问题

Task 12 的 production Executor 已经预编译 `ChunkPlan`，并复用 send/receive
workspace，但仍存在两个明确缺口：

1. `dequantize_reduce_tensors(..., output=...)` 的 mean 路径不能保证一次 kernel
   直接写出最终均值 shard，Task 12 报告中的 `fused_dequant_reduce` 为 `false`。
2. `pool_reduced_output=False` 是正确的安全默认值。若在 `Work` 完成时立即把
   ReducedShard 返回池中，调用方仍可能读取该 tensor，下一次通信会覆盖它。

## 方案比较

### 方案 A：默认池化最终 ReducedShard

优点是调用接口不变，steady-state allocator 开销最低。缺点是 CCDL 不知道
下游 optimizer 或 sharded consumer 何时读完结果；在 `Work.wait()` 后回收会
产生 use-after-reuse，依赖析构回收又不可预测。本方案不采用。

### 方案 B：只支持调用方提供 `out`

优点是所有权最清晰、热路径零分配、无需改变 Core `ReducedShard`。缺点是独立
用户必须自行维护 buffer，CCDL workspace pool 无法直接提供便利接口。本方案
作为主要性能接口，但不是唯一接口。

### 方案 C：安全默认值 + caller-owned output + 显式输出租约

默认调用继续返回独占、非池化 ReducedShard；高性能调用可传入 `out`；需要
CCDL 管理 buffer 时，调用方显式 acquire lease，并在消费 stream 上 release。
该方案同时保证安全性、独立库易用性和零分配能力，因此采用。

## 公共接口

### 融合 CUDA ABI

```python
inplace_dequantize_reduce_mean(
    inputs,
    output,
    group_size,
    topk,
    bit,
    quant_type,
    compact,
    divisor,
) -> bool
```

首个生产快路径支持：

- linear INT8；
- `group_size == 64`；
- `topk == 0`；
- contiguous FP16、BF16、FP32 输出；
- `sum` 和 `mean`，其中 mean 的 divisor 为 world size；
- 非整除原始 numel，由 ChunkPlan 产生的等长 padded shard。

不满足能力约束时返回 `False`，由 Python/CUDA codec 进入显式 fallback，并在
ExecutionInfo/ReducedShard metadata 中记录原因，不得伪装成 fused fast path。

### Executor 输出接口

```python
class CompressedReduceScatterExecutor:
    def run(
        self,
        tensor: object,
        *,
        out: object | None = None,
    ) -> CollectiveWork[ReducedShard]: ...

    def acquire_output(self) -> CudaOutputLease: ...
```

三种模式：

1. `run(tensor)`：默认安全模式。最终 shard 由调用方独占，不进入内部池。
2. `run(tensor, out=buffer)`：调用方拥有 buffer；CCDL 校验 shape、dtype、device、
   contiguous 和 alias 约束，融合 kernel 直接写入该 buffer。
3. `lease = executor.acquire_output()` 后
   `run(tensor, out=lease.buffer)`：buffer 来自 CCDL pool。调用方消费完成后调用
   `lease.release_after(tensor_or_event)`；lease 在对应 CUDA event ready 前不得复用。

Core `ReducedShard` 保持后端无关且不携带 CUDA lease，避免把资源生命周期混入
可序列化元数据模型。

## 编译期与数据面职责

编译阶段固定：

- ChunkPlan、shard shape 和 output workspace key；
- fused kernel capability；
- dtype、bit、group size、reduce divisor；
- fallback callable；
- workspace pool 和 completion manager。

`run()` 数据面只执行：

```text
validate tensor identity class
-> acquire/reuse send and receive payload workspaces
-> quant-pack destination chunks
-> compressed all-to-all
-> fused dequant-reduce-mean into out
-> record completion
-> return ReducedShard Work
```

数据面不得重新选择策略、创建逐 rank restored tensor，或构造完整梯度。

## 输出所有权与异步语义

- send/receive workspace 由 communication Work 持有到通信与融合 kernel 完成。
- 默认 ReducedShard output 永不自动回池。
- caller-owned `out` 的生命周期完全由调用方负责，Work 在完成前强引用它。
- pooled output lease 只有在显式 `release_after` 后进入 pending 队列，并由 CUDA
  event 查询确认消费 stream 完成后重新可用。
- 重复 release、跨 executor lease、shape/dtype/device 不匹配必须同步报错。
- 异常路径只释放内部 send/receive lease；不得擅自释放 caller-owned output。

## 错误处理与 fallback

- 非法 out buffer 在通信启动前失败。
- fused symbol 缺失或 capability 不满足时使用预分配 output 的非融合 fallback；
  mean 必须使用 in-place divide，禁止产生新的 output tensor。
- CUDA kernel launch 错误由 Work.wait()/Future 传播。
- fallback 原因写入 `ExecutionInfo.fallback_reason` 和
  `ReducedShard.metadata["fused_dequant_reduce_reason"]`。

## 测试与验收

### 单元测试

- TDD 覆盖 fused symbol 调用、参数和返回指针 identity；
- FP16/BF16/FP32，world size 1/2/3/4/5/8，非整除 numel；
- unsupported bit/group size/quant type/topk 的明确 fallback；
- caller-owned output 校验；
- output lease acquire、pending、event-ready reuse、double release 和异常路径；
- async Work 在 kernel 完成前持有所有输入与输出资源。

### CUDA 正确性

- fused 与 Task 12 fallback 输出在 INT8 阈值内一致；
- relative L2 不超过 0.02；
- non-finite 为 0；
- profiler 中生产快路径只有一次主 dequant-reduce-mean kernel，不出现逐 payload
  dequant kernel 和额外 mean kernel。

### A6000 性能门禁

使用 2 卡、4 卡，FP16 1/16/64 MiB，ABBA 顺序和至少 5 次独立运行：

- 基线为提交 `d7ab8e8` 的 Task 12 compiled ReducedShard；
- 候选使用同一 compressed transport、caller-owned/leased output 和 fused kernel；
- 第 2 至 100 次 steady-state 显式 allocator 调用为 0；
- 16/64 MiB 中位延迟不得慢于 Task 12；
- caller-owned 与 leased output 数值、指针和生命周期语义一致；
- 未满足门禁时保持 fallback，不将 fused 路径设为默认。

## 实施顺序

1. 完成 CUDA fused dequant-reduce-mean ABI 与 codec fallback。
2. 将 caller-owned `out` 接入 transport、Executor 和异步 Work。
3. 增加 `CudaOutputLease`，保持默认 output 不池化。
4. 接入 compiler capability 与 ExecutionInfo。
5. 完成本地、A6000 2/4 卡正确性、profiler、性能和显存门禁。

Task 12.1 全部通过后才进入 Task 13。
