# lowbit_comm 0.3.0 软件需求规格说明书

## 1. 文档信息

- 产品版本：0.3.0
- 变更等级：BREAKING — Major Architecture Refactor
- 状态：v2 重构需求基线
- 日期：2026-08-12
- 首要生产平台：NVIDIA GPU、CUDA、NCCL
- 验证平台：单机 2/4 卡及双机 4/8 卡 NVIDIA RTX A6000

## 2. 产品目标

lowbit_comm 是独立、GPU 优先、编译式的低比特通信库。它在通信受限且压缩收益大于
量化、同步与恢复开销的场景中提供可重复的端到端训练加速；当证据不足或压缩不适用
时，`auto` 必须安全选择 Native NCCL。

产品不承诺所有模型、消息规模和 world size 获得固定加速，也不承诺普遍达到 2 倍。
数值正确性、跨 rank 一致性、训练收敛、异步资源安全和可诊断性不可牺牲。

## 3. 范围与边界

### 3.1 0.3.0 必须交付

- 强类型 Semantic IR 与 Backend Lowered IR；
- `compile-once / run-many` 统一接口；
- CUDA/NCCL Backend 与无设备依赖的 Reference Backend；
- Native collective、量化 collective 与量化 P2P；
- Quantized FullTensor 与 Quantized ReducedShard；
- INT8 生产路径和 INT4 显式实验路径；
- DDP FullTensor Adapter 与 Sharded Adapter；
- 梯度 EF 与参数差值 EF 的独立状态域；
- Work、Event、WorkspaceLease 的统一完成语义；
- 显式严格策略及实测证据驱动的 `auto`；
- 动态 shape 的版本化固定长度 metadata packet；
- 功能、数值、分布式、性能与 time-to-quality 门禁。

### 3.2 Core 不负责

- 模型、数据加载与训练任务调度；
- 优化器、FP32 master weights、checkpoint 或 loss scaling；
- collective 已提交后的透明算法切换；
- 完整替代 NCCL/HCCL；
- 使用长期 Git 分支维护不同硬件 Backend。

训练状态由 Adapter 或上层 Engine 拥有；Core 只验证、编译和执行通信程序。

## 4. 功能需求

### FR-001 强类型通信程序

`CommunicationProgram` 必须分别声明 `operation`、`output`、`wire` 和 `algorithm`。
最终输出 dtype 不得推导 wire dtype。新公共接口不得包含 `restore_mode`。

### FR-002 两级 IR

Semantic IR 不得包含 Torch tensor、process group、CUDA stream 或 workspace。Backend
lowering 才绑定设备操作、拓扑和运行资源。

### FR-003 Verifier

编译前必须拒绝：

- ReducedShard 配合必须完整恢复的算法；
- QuantizedWire 缺少量化 schema；
- FullTensor 缺少跨 rank 一致输出阶段；
- ReducedShard 存在隐式 full all-gather；
- QuantizedWire lowering 中存在未记录的 FP collective；
- Backend 不支持所请求 dtype、bit、world size、EF 域或异步契约。

### FR-004 统一编译接口

```python
executable = lowbit_comm.compile(program, context, bindings=bindings)
work = executable.run(tensor, out=None)
result = work.wait()
```

编译阶段完成验证、策略与拓扑选择、融合、workspace 规划和 Backend 绑定。

### FR-005 Quantized FullTensor

正常压缩路径必须为：

```text
quantized reduce-scatter
-> fused dequant-reduce-mean-requantize
-> quantized all-gather
-> each-rank gathered-dequant-writeback
-> identical FullTensor
```

两个跨 rank 阶段均传输量化 payload。最后每个 rank 在本地执行一次完整恢复，不再次
reduce。FP shard all-gather 只能作为显式 FullPrecisionWire 或可观察 fallback。

### FR-006 Quantized ReducedShard

ReducedShard 路径在本 rank 产生全局归约结果的确定分片后直接返回，不得执行最终
full all-gather。元数据必须包含逻辑范围、padding、reduction、dtype 和 layout version。

### FR-007 通信原语

必须支持 all-reduce、all-gather、reduce-scatter、all-to-all、broadcast、reduce、
gather、scatter、barrier，以及 send/recv/isend/irecv。显式 Native 请求直接调用后端
原生 collective。

### FR-008 量化能力

CUDA Backend 必须支持 FP16/BF16/FP32 输入、INT8 group-wise linear quantization、
compact payload 与 caller-owned output。INT4、stochastic、top-k 在通过性能和收敛门禁
前只能显式使用，不进入 `auto`。

### FR-009 Error Feedback

必须区分 GradientErrorFeedback 与 ParameterDeltaErrorFeedback。状态键至少包含 tensor
identity、layout generation、shape、dtype、world size 与 compression schema。策略未
实现相应 EF 域时必须编译失败或由 `auto` 产生显式 fallback。

### FR-010 异步完成

`Work` 完成同时表示 collective、GPU 后处理、必要 EF 更新以及输出对 consumer stream
可见。Future 不得早于最终 event 完成；`wait()` 不得临时启动未调度的主要计算。

### FR-011 Workspace 所有权

CompiledExecutable 拥有可选 pool。每次执行获取 WorkspaceLease，完成 event ready 后
才可回收。caller-owned output 和交给用户的独占结果不得自动入池或被覆盖。

### FR-012 策略语义

显式算法严格执行，不支持则编译失败。只有 `auto` 可根据能力与版本化实测证据选择
压缩或 Native。执行热路径禁止 registry lookup、capability probe 和 fallback 解析。

### FR-013 动态 metadata

动态 shape 使用版本化固定长度 metadata packet，包含协议版本、shape、dtype、quant
schema、payload length、layout generation 与 flags。静态 bucket 不得每步调用
`all_gather_object`。

### FR-014 Adapter

DDP Adapter 拥有 bucket/AMP 生命周期与 Gradient EF；Sharded Adapter 直接消费
ReducedShard。qWD 属于 Sharded Adapter 参数同步算法，不是 Core collective strategy。

### FR-015 可观测性

ExecutionInfo 必须记录 requested/effective algorithm、wire、output、通信字节、融合阶段、
workspace、fallback 原因和 benchmark evidence ID。

## 5. 非功能需求

### NFR-001 性能

- 稳态热路径禁止策略字符串解析、注册查询、能力探测、进程组/stream 创建、workspace
  shape 规划和无条件 CPU synchronize；
- 已验证生产快路径相对 0.2.x 最终基线中位吞吐退化不得超过 2%；
- 代表性通信受限训练的目标中位端到端吞吐提升至少 10%；
- 不适合压缩的计算受限场景通过 Native 回退将退化控制在 2% 内。

### NFR-002 正确性

- FullTensor 所有 rank 输出在定义容差内一致；
- ReducedShard 与全精度 reference 的对应逻辑分片一致；
- NaN/Inf、padding、非整除 shape、空 shard 与 world-size 变化必须显式处理；
- AMP overflow、checkpoint restore 与 layout rebuild 不得错误提交或复用旧状态。

### NFR-003 通用性

IR、metadata、workspace key 与算法不得写死 2/4/8 卡。这些规模仅用于 0.3.0 验证；
设计必须支持任意合法 world size，并为 8 机 64 卡保留分层 lowering 能力。

### NFR-004 依赖方向

Core 不导入 Torch/CUDA/Ascend/Adapter；Backend 不反向导入 Adapter 或高层 API；
Backend 之间不交叉导入；新生产代码不依赖旧 `ccdl_comm` Python 控制面。

### NFR-005 发布洁净度

0.3.0 必须从洁净 clone 构建 wheel、安装并测试。仓库不得包含凭据、签名 URL、模型、
数据集、本地绝对路径、构建缓存或未经批准的大型 benchmark 产物。

## 6. 验收

### 6.1 硬件矩阵

- 单机 2 卡 A6000；
- 单机 4 卡 A6000；
- 双机 4 卡 A6000；
- 双机 8 卡 A6000。

### 6.2 工作负载

- collective 与 Kernel 微基准；
- 约 6291 万参数 synthetic MLP；
- 21G 真实数据集；
- 至少一个公开可复现模型/数据集；
- 长程 time-to-quality 收敛任务。

Native、0.2.x 最终基线和 0.3.0 在相同 GPU、容器、dtype、batch、步数下交替至少三次，
报告中位数与离散度。只有精确匹配 GPU、拓扑、world size、dtype、bucket、wire、output
和 EF 的证据才能进入 `auto`。

## 7. 发布与分支集成

1. 先将 `codex/correctness-kernel-hardening` 作为 0.2.x 最终基线以独立 PR 合入 main；
2. 为该合并点建立不可变性能基线 tag；
3. 将 `codex/v0.3.0-major-refactor` 更新到该 main 基线之上；
4. 0.3.0 PR 只包含大重构的需求、架构、实现、迁移与删除；
5. 包名统一为 `lowbit_comm`、版本为 `0.3.0`，发布说明标注 BREAKING；
6. 必选功能和门禁全部 READY 后才删除旧控制面并合并发布。
