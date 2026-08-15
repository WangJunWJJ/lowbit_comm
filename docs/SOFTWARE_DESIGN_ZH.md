# lowbit_comm 0.4.0 软件架构设计说明书

## 1. 当前架构决策

v0.4.0 采用编译期策略层、不可变 ExecutionPlan 和直接 Backend 热路径。Phase 1 的目标
是把数学语义、策略、证据、能力、计划和完成状态分层，并在没有 torch/CUDA 的环境中
完成契约验证；设备生产实现推迟到 Phase 2。

该版本不兼容已删除的 v0.3 Python API，也不提供兼容 facade。旧 API 名称、Backend
loader、Registry、EvidenceStore、Compiler 和扩展入口都不进入顶层公开面。

## 2. 分层和依赖

```text
stable api types
    -> compile_communicator
        -> Compiler (compile time only)
            -> Evidence + Registry
                -> production Backend.lower
                    -> immutable ExecutionPlan
                        -> BackendPlan.execute (steady-state only)

ReferenceBackend -> group oracle tests only
```

目录职责：

```text
lowbit_comm/
├── api/                 # Intent、Policy、Result、Communicator facade
├── core/                # Error、CompilationContext、ExecutionPlan、signature
├── compiler/            # Registry、Evidence、Compiler
├── backends/
│   ├── protocols.py     # production capability/lowering protocol
│   └── reference/       # oracle-only group execution
└── runtime/             # Work 与 error-feedback 事务状态机
```

依赖只允许朝执行细节方向流动。顶层导入不加载 torch、`lowbit_comm._C`、设备 Backend
或 Reference oracle。Reference 不实现 production Backend protocol，因此不能进入
Registry/Compiler/facade。

## 3. 稳定语义模型

### 3.1 Intent

`TensorSpec` 固化 dtype 和 shape；`ShapeFamily` 固化可接受 numel 上界与 alignment。
`CommunicationIntent` 组合 ReductionOp、OutputSemantics、CompletionMode、world size 和
rank。所有签名对象 frozen/slotted，并在 `__post_init__` 使用精确类型验证。

### 3.2 Policy 与 Strategy

`StrategySpec` 的 compression、collective、topology、accumulation、EF、overlap 和
workspace 字段保持正交，并拒绝矛盾组合。

编译优先语义为：

```text
Explicit exact strategy
    > Auto exact Production-Auto evidence within constraints
    > Auto native fallback
```

`NativePolicy` 是独立直达路径。Explicit 缺少 capability 时抛出 CapabilityError；Auto
只有精确证据、约束、资源和 capability 同时成立才选择压缩候选，否则固化 Native
fallback。执行阶段看不到 Policy。

### 3.3 Result

`FullTensorResult[T]` 包含完整聚合值。`ReducedShardResult[T]` 组合值和
`ReducedShardMetadata`；metadata 明确 global shape、线性 offset、valid/padded length
与 owner rank。两类结果不通过 flag 合并，避免 consumer 混淆所有权。

## 4. Registry 与 Backend protocol

`BackendCapability` 以 Backend ID、一个精确类型的完整不可变 `StrategySpec`、output、
world-size 范围、dtype 和 async 支持描述一个精确能力。它不复制 strategy 的部分字段；
`supports(intent, strategy)` 要求 capability 持有的 strategy 与请求 strategy 完整相等。
`BackendRegistry` 展开 Backend 声明的 capability，并以统一 canonical helper 编码
`StrategySpec` 的全部 dataclass 字段，按完整 key 稳定排序。重复 capability 或 Backend
身份冲突立即失败。诊断 world-size 枚举与精确候选查询分离，Compiler 只调用后者。

生产 `Backend.lower(intent, strategy)` 返回一个结构化 `BackendPlan`；其执行接口是：

```python
def execute(self, value: object) -> CommunicationWork[object]: ...
```

Phase 1 没有具体 production Backend。ReferenceBackend 只提供 `compile_group()` 和
all-rank oracle 执行，没有 `capabilities()`、`lower()` 或 rank-local `execute()`；这使
测试 oracle 不可能被误注册为生产实现。

## 5. Evidence 和 Compiler

Evidence 使用规范化、可哈希的完整 key 绑定 environment、intent、strategy、node count、
workload class 和 bucket range。当前 schema-v2 的 `EvidenceMetrics` 要求全部百分比为
exact finite float，并持久化普通通信收益与暴露通信收益；`EvidenceRecord` 保存调用方
声明的状态，并在构造和每个信任边界重新要求它等于 `derive_evidence_status(metrics)`。

状态导出先运行通信门。通信收益小于 -2% 时为 Rejected；通信收益不足 5% 且暴露通信
收益不大于 0 时为 Experimental；其余情况才获得 Long-Test 准入。在准入之后，端到端
门未达到 Recommended 条件时保持 Long-Test，达到 Recommended 条件时才晋级
Recommended，且只有端到端收益至少 10%、质量与收敛门通过、至少 3 个 seed、跨
workload 复现且最差运行不低于 -2% 时才晋级 Production-Auto。因此端到端结果不能
绕过通信回归或通信准入门。

schema-v1 使用独立的 legacy metrics/record 强类型表示，保留原始字段和声明状态用于
历史诊断，且其既有 key/record 指纹编码保持不变。`from_request` 只生成 schema-v2；
EvidenceStore 可保存并重验 v1，但 Production-Auto lookup 和 Compiler 只信任完整、
可重新验证的 schema-v2。系统不使用 Optional 字段冒充 v2，也不静默迁移 v1。

Compiler pipeline 为：

```text
validate exact intent/policy/context
-> compute deterministic cache key
-> resolve Native / Explicit / Auto
-> validate strategy against output and resource context
-> query exact Registry candidates
-> Backend.lower
-> bind provenance and evidence fingerprint
-> construct immutable ExecutionPlan
-> cache by complete signature
```

Registry generation 和 Evidence generation 都进入 cache key。计划签名包括 intent、
strategy、context、Backend、origin 和 evidence fingerprint，避免跨环境错误复用。

## 6. Facade

稳定入口为：

```python
@dataclass(frozen=True, slots=True)
class CompiledCommunicator:
    plan: ExecutionPlan

    def execute(self, value: object) -> CommunicationWork[object]:
        return self.plan.backend_plan.execute(value)
```

`compile_communicator()` 在调用 Compiler 前验证 exact CommunicationIntent、三种 exact
Policy、exact CompilationContext，以及 compiler 是否提供可调用 `compile()`。compiler
保持结构化边界，允许测试 double 和未来符合该调用契约的编译实现，但 Registry 和
Evidence 不因此成为公开参数。

Compiler 调用恰好一次。返回后，facade 要求 exact ExecutionPlan 和可调用的 Backend
plan `execute()`，使结构错误在 compile 边界明确失败。构造成功后不再复制或验证计划。

热路径严格等价于上面的一行委派：不做 tensor 验证/复制，不读取 Policy，不访问
Compiler/Registry/Evidence，不 fallback，不执行 all-rank gather，也不识别 Reference
group plan。Backend 返回的 CompletedWork 或 FailedWork 保持对象身份原样交给调用方。

## 7. Work 与事务状态

`CommunicationWork[T]` 是结构化完成协议。`CompletedWork` 已完成并稳定发布原值；
`FailedWork` 已进入失败终态，`wait()` 和 `result()` 都抛出同一个 ExecutionError 实例。

error-feedback 状态机将一次更新拆为 prepare 和 commit/abort。只有对应 token 的成功
执行可以提交 prepared residual。abort 保留原始失败并不发布候选状态；重复或过期 token
不能改变已提交状态。设备 event/workspace 生命周期属于后续 Backend 实现。

## 8. Reference 数值 oracle

ReferenceBackend 对一个完整 rank-value tuple 做确定性归约。FullTensor 为每个 rank 创建
独立结果对象；ReducedShard 使用确定 offset 和 padding 拆分全局归约值。SUM/MEAN、
多维 numel、非整除 shape、rank 多于元素、零元素、非有限输入和归约溢出都有契约测试。

Reference 只验证语义，不模拟 NCCL、dtype rounding、设备异步、压缩 wire 或性能，不能
作为训练 Backend 或性能证据来源。

## 9. 公开导出和导入安全

顶层 `lowbit_comm.__all__` 与 `lowbit_comm.api.__all__` 使用相同精确集合：

```text
AccumulationDType, AutoConstraints, AutoPolicy, CollectiveKind,
CommunicationIntent, CommunicationWork, CompilationContext,
CompiledCommunicator, CompletionMode, CompressionKind, ExplicitPolicy,
FullTensorResult, NativePolicy, OutputSemantics, ReducedShardMetadata,
ReducedShardResult, ReductionOp, ShapeFamily, StrategySpec, TensorSpec,
TopologyKind, compile_communicator
```

其中枚举和构造类型足以描述 Intent/Strategy/Result；Compiler、ExecutionPlan、PlanOrigin、
Registry、EvidenceStore、Backend protocol/loader、ReferenceBackend、Completed/FailedWork、
`_C` 和旧 API 都是内部或测试表面。

导入链只触达纯 Python 标准库模块。隔离测试用 meta-path finder 主动拒绝 torch 和
`lowbit_comm._C`，并从包含中文的绝对工作树路径导入，以验证 CPU-only 安全导入。

## 10. Phase 1 架构门禁

- 根 `lowbit_comm/` 和根 `csrc/` 是唯一活动源码位置；
- 包版本来源只有项目元数据中的 0.4.0.dev0；
- 所有稳定类型的不可变性和非法组合都有 unit contract；
- Compiler 的 Explicit/Auto/Native、cache 和 signature 行为确定；
- facade compile-once、execute direct delegation 和 failure identity 通过 contract；
- Reference 明确不可注册为生产 Backend；
- 顶层精确导出和 isolated safe import 通过；
- 完整 pytest、Ruff、compileall 和仓库文档治理通过；
- `docs/` 只包含两份正式总文档，过程计划和任务报告不进入提交。
