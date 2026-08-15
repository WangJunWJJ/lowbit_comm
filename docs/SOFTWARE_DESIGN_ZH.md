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
Core 提供一个不导入 API、Compiler 或 Evidence 类型的安全调用器：它先要求 exact class，
再调用由类型归属模块传入的可信 unbound invariant 函数（例如
`TensorSpec.__post_init__(tensor)`），保留原 `CompileError`，并把其他 `Exception`
以 cause 规范化为稳定 `CompileError`。对象图递归规则仍由归属模块维护：Intent 重验
TensorSpec/ShapeFamily，Policy 重验 StrategySpec/AutoConstraints，CompilationContext
重验 EnvironmentFingerprint，ExecutionPlan 重验 intent/strategy 和 BackendPlan 结构。
因此不存在动态 `obj.__post_init__()` 调用，也没有 Core helper 对上层类型的反向依赖。

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
fallback。Auto 按 schema 和完整 evidence dimensions 的稳定顺序遍历全部 exact
Production-Auto 记录；单条不满足 constraints、CompilationContext 或 Registry
exact capability 时继续下一条。只对最终选中的 Backend 调用 `lower()`，且不
调用 Registry 的诊断枚举 API；全部候选失败才 Native fallback。执行阶段看不到 Policy。
最终策略无论来自 Native、Explicit、Production-Auto 或 Auto Native-fallback，都进入
同一个 bound-lowering 边界。该边界原样重抛同一 `LowbitCommError` 对象；其他普通异常
以原异常为 cause 包装为 `CompileError`。返回 plan 仍由 `ExecutionPlan` 完成静态协议
验证，cache hit 不再次 lowering。
`AutoPolicy.constraints` 使用 per-instance default factory；即使测试或 hostile caller
绕过 frozen 约束伪造一个 constraints 对象，也不会污染后续新 Policy 的默认图。

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

注册在 mutation 之前复用 Core 的静态 member/callable resolver。`backend_id` 必须由
class attribute、instance dict 或已初始化 slot 提供非空 exact `str`；`capabilities`
与 `lower` 可由普通 method、exact `staticmethod`/`classmethod`、instance-dict callable
或 slot callable 提供。解析不调用用户 `__getattribute__`/`__getattr__`，也不绑定
property、动态或用户自定义 descriptor（包括 callable descriptor 和内置 descriptor
的恶意 subclass）。静态解析 ID 后，Registry 从已提交 `_BackendEntry` 扫描推导该 ID
的唯一 owner，只用 `is` 比较对象身份；不同对象的 ID 冲突或同 ID entries 的内部 owner
漂移均稳定 fail closed，且发生在 contender 的 `capabilities`/`lower` 解析与调用之前。
同一 owner 对象可分批追加非重叠 capability，不引入平行 owner map。Registry 随后验证
`capabilities` 与 `lower`，并恰好调用一次
`capabilities()`；返回值必须是非空 exact tuple，元素必须是经字段不变量重新校验的
exact `BackendCapability`。每个返回图随后通过 trusted snapshot helper 以显式命名字段
重建：嵌套 `StrategySpec` 独立构造，`supported_dtypes` 通过逐项生成器构造新的
`frozenset`，不可变枚举和标量按值保留；不使用 `copy`/`deepcopy`/`repr` 或输入对象的
动态方法。显式构造的字段分类由 dataclass completeness guard 约束，字段漂移时 fail
closed。全部可信 snapshot 在临时映射中通过 ID、一致性和重复检查后才一次性写入；首次
成功写入同时建立 ID 身份所有权，每次成功注册只增长一次 generation；
空、畸形、伪造或冲突批次不产生 owner 占位。任一失败保持 entries 与 generation 不变，
且注册阶段从不执行 `lower()`。
结构或返回值错误规范化为 `CompileError`，重复键继续使用 `CapabilityError`。

Registry 的内部 `_BackendEntry` 是 frozen、slotted 的单一事实源，同时保存可信 capability snapshot、
Backend 身份 owner 和上述静态解析得到的 bound `lower` callable；同一 Backend 的协议成员各解析
一次，`capabilities()` 也只调用一次，所有 capability entry 共享该 callable。不存在
平行 lower map。诊断候选和 `resolve_exact()` 仍投影为 `(capability, backend)` 二元
Backend match，但每个公开或诊断出口都从内部 snapshot 显式重建全新的 capability、
嵌套 strategy 与容器，绝不返回内部实例。Compiler 通过 compiler-only exact resolution
按调用方 capability 的完整值 key 定位 entry，再取得另一份独立 capability 投影和
`bound_lower`，此后绝不再次访问 `backend.lower`。Registry 在 owner 扫描与每条查询路径
重新校验内部 entry 的 dict key 等于其 snapshot key；任何内部漂移统一抛出稳定
`CompileError`，不进入候选或 lowering。

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

schema-v2 使用一个规范、类型感知的 `intent_signature` 维度绑定所有
collective-shared intent 语义：tensor dtype 和完整 shape、ShapeFamily 的 max-numel
与 alignment、reduction、output、completion 和 world size。编码不使用
`repr`，并通过 dataclass 字段分类在未来字段未明确定义为 shared 或 local
时 fail closed；TensorSpec 和 ShapeFamily 的所有 dataclass 字段都反射进稳定
编码。本地 rank 是唯一明确排除项，使同一 collective 的所有 rank 生成相同
证据 key。另外保留 exact shape、ShapeFamily、reduction 和 completion 等可读维度，
并将它们纳入环境冲突检查。

schema-v2 的 `strategy_signature()` 输出是兼容既有 key 的 legacy partial token，而非
完整 exact strategy 编码。`StrategySpec` 的每个当前字段通过显式 field-to-dimensions
classifier 归入该 token 或 topology、bit width、group size、error feedback、overlap、
wire size 等现有维度；分类必须非空且只能引用 schema-v2 已有维度。Core 的共享
dataclass coverage guard 反射真实字段集合并与 classifier keys 精确比较，任何新增、
删除、改名、遗漏或未知字段都抛 `CompileError` 并要求 schema bump，不能自动改变 v2。

状态导出先运行通信门。通信收益小于 -2% 时为 Rejected；通信收益不足 5% 且暴露通信
收益不大于 0 时为 Experimental；其余情况才获得 Long-Test 准入。在准入之后，端到端
门未达到 Recommended 条件时保持 Long-Test，达到 Recommended 条件时才晋级
Recommended，且只有端到端收益至少 10%、质量与收敛门通过、至少 3 个 seed、跨
workload 复现且最差运行不低于 -2% 时才晋级 Production-Auto。因此端到端结果不能
绕过通信回归或通信准入门。

schema-v1 使用独立的 legacy metrics/record 强类型表示，保留原始字段和声明状态用于
历史诊断，且其既有 key/record 指纹编码保持不变。`from_request` 只生成 schema-v2；
EvidenceStore 可保存并重验 v1，但 Production-Auto lookup 和 Compiler 只信任完整、
可重新验证的 schema-v2。v1 与 v2 各自使用精确的必需维度集合：旧 v1 key
仍可读取，缺少当前 intent 签名或任一必需语义维度的 v2 key 被拒绝。系统不
使用 Optional 字段冒充 v2，也不静默迁移 v1。
Evidence 的 current/legacy record validator 是各自对象图的唯一事实源，并通过共享安全
调用器递归重验 exact key、完整 StrategySpec、对应 metrics 与 record 语义。Compiler
继续逐条丢弃伪造 current candidate、忽略诊断型 legacy candidate，并可选择后续有效
current record 或耗尽后 Native fallback；但 EvidenceStore 自身的 `records` 必须仍为
exact tuple，容器边界畸形立即抛 `CompileError`，不泄漏迭代异常。

Compiler pipeline 为：

```text
fresh-validate complete exact intent/policy/context graphs
-> compute deterministic cache key
-> resolve Native / Explicit / Auto
-> validate strategy against output and resource context
-> query exact Registry candidates
-> Backend.lower
-> bind provenance and evidence fingerprint
-> bind BackendPlan.execute statically
-> store private immutable cache entry by complete signature
-> project a fresh public ExecutionPlan and bound adapter
```

Registry generation 和 Evidence generation 都进入 cache key。计划签名包括 intent、
strategy、context、Backend、origin 和 evidence fingerprint，避免跨环境错误复用。
Compiler 的全部手写 canonicalizer 在读取字段前复用同一个 exact-type coverage guard，
覆盖 CommunicationIntent 及其 nested tensor/shape family、StrategySpec、AutoConstraints
与 policy wrapper、CompilationContext 与 environment、EvidenceKey、current/legacy record
及 metrics。guard 只做字段集合 preflight，不写入 payload，所以现有 JSON、排序、v1
golden 与 v2 fingerprint 不变。Compiler cache 的 intent 编码继续包含 rank；Evidence 的
collective-shared `intent_signature` 才排除 rank。
上述 caller 图验证严格早于 Evidence generation 和 cache lookup，所以畸形请求不会进入
Auto candidate 的可丢弃 `CompileError` 区域，也不会读取 cache、遍历 Evidence、查询
Registry 或执行 lower。

Compiler 在 caller graph preflight 后以显式字段构造 trusted intent、policy 和 context
快照；tuple/frozenset 容器逐项重建。策略选择完成后同样重建完整 `StrategySpec`，并为
Registry/lower、内部 cache entry 和公开投影使用彼此独立的语义图。Native 与 fallback
不再引用 module singleton；`_canonical_native_strategy()` 每次构造 fresh exact value。

cache value 是 private frozen/slotted `_CachedPlanEntry`，而不是 `ExecutionPlan`。它保存
canonical cache key、未暴露的 intent/strategy、provenance/signature/evidence、opaque
BackendPlan、运行期 identity token 与编译期静态解析的 execute callable。hit 先通过可信
exact-class validator 重验 entry 和 BackendPlan 静态结构，再核对 key、请求与 policy
语义、origin/evidence，并用当前 trusted context 重算 signature；失败抛 `CompileError`，
不执行 opaque plan。合法 hit 直接投影，不查询 Registry、不重新 lowering。

每次 miss/hit 都创建 fresh `ExecutionPlan`、intent/tensor/shape-family、strategy 和
`_BoundBackendPlan`。该 adapter 的 `execute()` 只有保存 callable 的单行调用，因此 facade
热路径不增加 cache/Registry lookup 或 validation，并原样传播 Work。CCDL 隔离公开 wrapper、
adapter 和语义图，但不声称深拷贝 BackendPlan 内的设备资源或 Backend-owned mutable state；
原始 opaque 对象与 bound callable 明确属于已注册 Backend 信任域。内部 cache 持有它们但
从不直接暴露。

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
Policy、exact CompilationContext，并静态解析 compiler 的可调用 `compile()`，避免通过
动态属性查找触发 descriptor。compiler 保持结构化边界，允许测试 double 和未来符合该
调用契约的编译实现，但 Registry 和 Evidence 不因此成为公开参数。其返回值必须是
exact `ExecutionPlan`，并在 request semantics 检查前直接重跑
可信 exact-class validator，递归重验 ExecutionPlan 自身、intent/TensorSpec/ShapeFamily、
strategy 和 BackendPlan 结构，从而拒绝构造后被伪造的空 Backend ID、空 signature、
值相等但类型非法的嵌套字段或失效 BackendPlan。Facade 对 caller intent/policy/context
使用与正式 Compiler 相同的完整图 validator，并在解析/调用 structural compiler 前完成。
该检查只发生在 compile boundary，不进入 `execute()` 热路径。

解析成功后，Compiler 恰好调用一次。返回计划必须是 exact ExecutionPlan，Backend plan
必须提供结构上可调用的 `execute()`；`plan.intent` 必须是 exact CommunicationIntent，
且按值等于请求 intent；strategy 和 origin 也必须分别是 exact StrategySpec 和
PlanOrigin。facade 随后执行以下 Policy 后置条件：

- NativePolicy 只接受 NATIVE origin、规范 Native strategy 和空 evidence fingerprint；
- ExplicitPolicy 只接受 EXPLICIT origin、按值等于显式请求的 strategy 和空 evidence
  fingerprint；
- AutoPolicy 接受满足全部 constraints、具有非空 evidence fingerprint 的 AUTO 计划，
  或接受 NATIVE_FALLBACK origin、规范 Native strategy 和空 evidence fingerprint。

任一结构或语义后置条件失败都在发布 communicator 前抛出 CompileError，并且不执行
Backend plan。构造成功后不再复制或重新验证计划。

热路径严格等价于上面的一行委派：不做 tensor 验证/复制，不读取 Policy，不访问
Compiler/Registry/Evidence，不 fallback，不执行 all-rank gather，也不识别 Reference
group plan。Backend 返回的 CompletedWork 或 FailedWork 保持对象身份原样交给调用方。

## 7. Work 与事务状态

`CommunicationWork[T]` 是结构化完成协议。`CompletedWork` 已完成并稳定发布原值；
`FailedWork` 已进入失败终态，`wait()` 和 `result()` 都抛出同一个 ExecutionError 实例。

Phase 1 的 `ErrorFeedbackTransaction` 是 caller-driven 状态机：CREATED 可以 prepare 后
进入 PREPARED，再由调用方 commit 为 COMMITTED，或者从 CREATED/PREPARED abort 为
ABORTED。候选 residual 只有在 COMMITTED 才可见；其余状态继续发布旧 residual。非法
转换抛出 ExecutionError，abort 后的非法 prepare/commit 以原始失败为 cause。

commit 表示调用方断言通信已经成功；状态机不接收也不验证 CommunicationWork、completion
event 或 launch token。launch/completion token 绑定、stale completion/event 拒绝、CUDA
stream ordering 和设备 event/workspace 生命周期属于 Phase 2。

## 8. Reference 数值 oracle

ReferenceBackend 对一个完整 rank-value tuple 做确定性归约。FullTensor 为每个 rank 创建
独立结果对象；ReducedShard 使用确定 offset 和 padding 拆分全局归约值。SUM/MEAN、
多维 numel、非整除 shape、rank 多于元素、零元素、非有限输入和归约溢出都有契约测试。

Reference 只验证语义，不模拟 NCCL、dtype rounding、设备异步、压缩 wire 或性能，不能
作为训练 Backend 或性能证据来源。

## 9. 公开导出和导入安全

顶层 `lowbit_comm.__all__` 与 `lowbit_comm.api.__all__` 使用相同的 26 项精确集合：

```text
AccumulationDType, AutoConstraints, AutoPolicy, CapabilityError,
CollectiveKind, CommunicationIntent, CommunicationWork, CompilationContext,
CompileError, CompiledCommunicator, CompletionMode, CompressionKind,
ExplicitPolicy, ExecutionError, FullTensorResult, LowbitCommError,
NativePolicy, OutputSemantics, ReducedShardMetadata, ReducedShardResult,
ReductionOp, ShapeFamily, StrategySpec, TensorSpec, TopologyKind,
compile_communicator
```

四个异常导出与 `lowbit_comm.core.errors` 中的类保持对象身份：LowbitCommError 直接继承
Exception，CompileError、CapabilityError 和 ExecutionError 直接继承 LowbitCommError。
枚举和构造类型足以描述 Intent/Strategy/Result；Compiler、ExecutionPlan、PlanOrigin、
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
