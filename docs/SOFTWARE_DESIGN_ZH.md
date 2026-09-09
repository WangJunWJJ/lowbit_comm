# lowbit_comm 0.4.0 软件架构设计说明书

## 1. 当前架构决策

v0.4.0 采用编译期策略层、不可变 ExecutionPlan 和直接 Backend 热路径。Phase 1 已把
数学语义、策略、证据、能力、计划和完成状态分层，并在没有 torch/CUDA 的环境中完成
契约验证。当前 Phase 2 已接入显式 c10d ProcessGroup 驱动的 CUDA FullTensor 与
ReducedShard 执行链，并通过独立 `lowbit_comm.experimental` 层提供 fail-closed
RSAG/qWD wheel adapter；该层不改变稳定 API 或 Production-Auto capability。

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

Facade 对正式 Compiler 的依赖只存在于 postponed type annotation 与 `TYPE_CHECKING` 分支；
导入 `lowbit_comm.api.communicator` 或顶层包不会运行时导入 Compiler、Registry 或 Evidence
实现模块。compile boundary 仍通过静态 callable resolver 接受正式 Compiler 与 structural
double，不把编译实现加入公开导出面。

目录职责：

```text
lowbit_comm/
├── _version.py         # package metadata 的唯一版本源
├── api/                 # Intent、Policy、Result、Communicator facade
├── core/                # Error、CompilationContext、ExecutionPlan、signature
├── compiler/            # Registry、Evidence、Compiler
├── backends/
│   ├── protocols.py     # production capability/lowering protocol
│   └── reference/       # oracle-only group execution
├── experimental/        # torch-lazy RSAG/qWD 证据、兼容性、状态与 plan adapter
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
`ReducedShardResult.__post_init__` 通过 Core trusted exact validator 直接重跑
`ReducedShardMetadata.__post_init__`，使 metadata 所有者仍是唯一不变量来源，
并将意外校验异常稳定为 CompileError。该校验不读取或限制泛型 value。
Reference 与 CUDA Native/INT8 FullTensor 必须发布相同 exact `FullTensorResult` envelope；
设备 Backend 不得因策略不同退化为裸 Tensor。ReducedShard 始终发布带 ownership metadata
的 `ReducedShardResult`，两者 consumer 不能互换。

## 4. Registry 与 Backend protocol

`BackendCapability` 以 Backend ID、一个精确类型的完整不可变 `StrategySpec`、output、
world-size 范围、dtype 和 async 支持描述一个精确能力。它不复制 strategy 的部分字段；
`supports(intent, strategy)` 是 backend-facing 信任边界，先通过 capability snapshot 与
Intent/Strategy 所有者 validator fresh 重验三个 exact 完整图，再调用不导出的纯
`_supports_request`；匹配要求 capability 持有的 strategy 与请求 strategy 完整相等。
`BackendRegistry` 展开 Backend 声明的 capability，并以统一 canonical helper 编码
`StrategySpec` 的全部 dataclass 字段，按完整 key 稳定排序。重复 capability 或 Backend
身份冲突立即失败。诊断 world-size 枚举与精确候选查询分离，Compiler 只调用后者。
精确候选查询在内部 entry 校验、排序或支持判断之前复用同一 Intent/Strategy owner
validators；畸形嵌套图稳定抛 `CompileError`，不进入 Registry 遍历。

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

Registry 的内部 `_BackendEntry` 是 exact `NamedTuple` 定义的 tuple-backed 单一事实源，
无 instance `__dict__` 或可设置 slot，同时保存可信 capability snapshot、
Backend 身份 owner 和上述静态解析得到的 bound `lower` callable；同一 Backend 的协议成员各解析
一次建立注册快照，`capabilities()` 也只调用一次，所有 capability entry 共享该
callable。查询时只用 Core static resolver 重验 saved callable 身份，不做动态读取或执行。不存在
平行 lower map。诊断候选和 `resolve_exact()` 仍投影为 `(capability, backend)` 二元
Backend match，但每个公开或诊断出口都从内部 snapshot 显式重建全新的 capability、
嵌套 strategy 与容器，绝不返回内部实例。Compiler 通过 compiler-only exact resolution
按调用方 capability 的完整值 key 定位 entry，再取得另一份独立 capability 投影和
`bound_lower`，此后绝不动态访问 `backend.lower`。Registry 在 owner 扫描与每条查询路径
重新校验 exact envelope、dict key/snapshot key、owner 的静态 backend ID，以及 saved/static
lower callable 绑定；任何内部漂移统一抛出稳定
`CompileError`，不进入候选或 lowering。

生产 `Backend.lower(intent, strategy)` 返回一个结构化 `BackendPlan`；其执行接口是：

```python
def execute(self, value: object) -> CommunicationWork[object]: ...
```

`CudaBackend` 是当前具体 production Backend。它在构造时持有调用方显式传入的 c10d
`ProcessGroup`，通过精确 capability/lowering 生成 rank-local FullTensor plan；不依赖
或替换默认进程组。ReferenceBackend 只提供 `compile_group()` 和 all-rank oracle 执行，
没有 `capabilities()`、`lower()` 或 rank-local `execute()`；这使测试 oracle 不可能被
误注册为生产实现。

## 5. Evidence 和 Compiler

Evidence 使用规范化、可哈希的完整 key 绑定 environment、intent、strategy、node count、
workload class 和 bucket range。当前 Compiler Evidence schema-v3 的 `EvidenceMetrics`
要求全部百分比为 exact finite float，并持久化普通通信收益与暴露通信收益；
`EvidenceRecord` 保存调用方声明的状态，并在构造和每个信任边界重新要求它等于
`derive_evidence_status(metrics)`。

schema-v3 使用一个规范、类型感知的 `intent_signature` 维度绑定所有
collective-shared intent 语义：tensor dtype 和完整 shape、ShapeFamily 的 max-numel
与 alignment、reduction、output、completion 和 world size。编码不使用
`repr`，并通过 dataclass 字段分类在未来字段未明确定义为 shared 或 local
时 fail closed；TensorSpec 和 ShapeFamily 的所有 dataclass 字段都反射进稳定
编码。本地 rank 是唯一明确排除项，使同一 collective 的所有 rank 生成相同
证据 key。另外保留 exact shape、ShapeFamily、reduction 和 completion 等可读维度，
并将它们纳入环境冲突检查。

schema-v3 的 `strategy_signature()` 把 `strategy_key()` 中 `StrategySpec` 的每个字段按
声明顺序编码为无空白 JSON；每项同时携带字段名、显式类型判别和值，不依赖 `repr`。
`StrategySpec` 的每个当前字段通过显式 field-to-dimensions classifier 归入 `strategy`
签名以及适用的 topology、bit width、group size、error feedback、overlap、wire size 等
独立维度；分类必须非空且只能引用 schema-v3 已有维度。Core 的共享
dataclass coverage guard 反射真实字段集合并与 classifier keys 精确比较，任何新增、
删除、改名、遗漏或未知字段都抛 `CompileError` 并要求 schema bump，不能自动改变 v3。

状态导出先运行通信门。通信收益小于 -2% 时为 Rejected；通信收益不足 5% 且暴露通信
收益不大于 0 时为 Experimental；其余情况才获得 Long-Test 准入。在准入之后，端到端
门未达到 Recommended 条件时保持 Long-Test，达到 Recommended 条件时才晋级
Recommended，且只有端到端收益至少 10%、质量与收敛门通过、至少 3 个 seed、跨
workload 复现且最差运行不低于 -2% 时才晋级 Production-Auto。因此端到端结果不能
绕过通信回归或通信准入门。

`EvidenceKey`、`EvidenceMetrics`、`EvidenceRecord` 和 `EvidenceStore` 只实现 schema-v3。
schema-v1/v2 key 与旧 record 表示在入口 fail closed，不在运行时保留兼容类、读取器、
迁移器、序列化器或诊断路径；历史数据由外部报告归档。Evidence record validator 是
对象图的唯一事实源，并通过共享安全调用器递归重验 exact key、完整 StrategySpec、
metrics 与 record 语义。Compiler 逐条丢弃构造后伪造的 candidate，并可选择后续有效
record 或耗尽后 Native fallback；但 EvidenceStore 自身的 `records` 必须仍为 exact
tuple，容器边界畸形立即抛 `CompileError`，不泄漏迭代异常。

`_normalize_records` 是 lookup、Compiler Auto 枚举和 Evidence generation 的唯一 use-time
pipeline。它先重验 exact store/tuple，再通过上述 owner validator 丢弃非法 record；
只用 fresh-valid exact EvidenceKey 中的 builtin int/str/tuple 值重建
canonical identity，不依赖 EvidenceKey 对象的可覆盖 hash/equality。同一 identity 有多条
时整组排除，其他唯一有效项按既有 key 顺序返回，所以事后伪造的重复 key
不会产生 first/last 顺序依赖。schema-v3 record 的 canonical payload 和 golden
fingerprint 独立覆盖完整 strategy graph 与全部 metrics。

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
与 policy wrapper、CompilationContext 与 environment、EvidenceKey、EvidenceRecord
及 metrics。guard 只做字段集合 preflight，不写入 payload，所以既有 intent/context
JSON 编码与排序不受影响；record fingerprint 使用 schema-v3 golden。Compiler cache 的
intent 编码继续包含 rank；Evidence 的
collective-shared `intent_signature` 才排除 rank。
上述 caller 图验证严格早于 Evidence generation 和 cache lookup，所以畸形请求不会进入
Auto candidate 的可丢弃 `CompileError` 区域，也不会读取 cache、遍历 Evidence、查询
Registry 或执行 lower。

Compiler 在 caller graph preflight 后以显式字段构造 trusted intent、policy 和 context
快照；tuple/frozenset 容器逐项重建。策略选择完成后同样重建完整 `StrategySpec`，并为
Registry/lower、内部 cache entry 和公开投影使用彼此独立的语义图。Native 与 fallback
不再引用 module singleton；`_canonical_native_strategy()` 每次构造 fresh exact value。

cache value 是 private exact `NamedTuple` `_CachedPlanEntry`，而不是 `ExecutionPlan`。真正的
tuple storage 无 instance `__dict__` 或可设置 slot，因此不能用 `object.__setattr__` 原位替换
opaque plan/identity/execute 执行锚。它保存
canonical cache key、未暴露的 intent/strategy、provenance/signature/evidence、opaque
BackendPlan、运行期 identity token 与编译期静态解析的 execute callable。hit 先通过可信
exact tuple validator 重验 entry 和 BackendPlan 静态结构/执行绑定，再核对 key、请求与 policy
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

Phase 2 的 native runtime 使用 `CudaWork`、`LaunchToken`、`WorkspacePool` 和
move-only `WorkspaceLease`。`CudaExecutor` 为每个实例分配唯一 plan id，并以原子序列号
标识每次 launch；launch 在当前 CUDA stream 上记录 event。`CudaWork.is_completed()`
只执行 event query，`wait()` 通过原子 owner + condition variable 保证并发调用者只进行
一次同步，随后缓存终态；析构路径先安全清理 event，再释放 lease。pool 用实际设备
`uint8` Tensor 承载 workspace。pool 按精确 byte size 保存已完成 lease 的 free buffer，
在容量预算内复用同一 storage；仍在飞行的 lease 不进入 free cache，容量不足时先逐出
free buffer。kernel、transport、event record/synchronize 失败会 poison pool 并 quarantine
对应 storage，后续请求稳定失败而不是复用可疑数据。

该 runtime 不接受 legacy Python callback/completion/query 对象，也不把这些对象保存在
异步热路径中。当前 token 负责 launch identity 与诊断；token 与 error-feedback 事务的
强绑定、stale completion 拒绝及多 stream 完整有序链仍未交付。

## 8. CUDA FullTensor 生产执行链

Python lowering 先根据 exact Intent/Strategy 生成 descriptor 和确定性 workspace layout，
再把 descriptor、workspace pool 与显式 ProcessGroup 交给原生
`create_fulltensor_plan()`。原生工厂再次逐字段校验 dtype、reduction、compression、
collective、numel、rank/world size、group size、payload/padding/gather/workspace bytes，
并核对 ProcessGroup 的 rank 和 size。当前 capability 限定 FP16/BF16、SUM/MEAN、
FullTensor、2/4 rank；INT8 group size 限定 16/32/64。

Native 执行顺序是：

```text
rank-local FP tensor
-> ProcessGroup all-reduce(SUM)
-> optional in-place divide(world_size) for MEAN
-> record CUDA completion event
-> CudaWork
```

INT8 执行顺序是：

```text
rank-local FP tensor
-> compact quantize-pack(values + FP16/BF16 scale per group)
-> ProcessGroup direct base all-gather of one contiguous quantized payload
-> one fused dequant-reduce kernel over every rank payload
-> optional mean scale inside fused kernel
-> write FP16/BF16 output
-> record CUDA completion event
-> CudaWork holding the workspace lease
```

因此聚合输入确实是 INT8 payload，而不是先反量化再通信；完整 FP 输出只在所有 payload
到齐后由 fused kernel 写一次。尾部 group 通过 padding 处理，逻辑输出按原 numel
截断。workspace lease 在 Work 终态前不归还 pool，重复 execute 可复用相同 byte size 的
已释放 storage；稳态不再为每个 rank 构造接收 Tensor list。Python plan 构造时绑定一次
native execute callable 和可信 Intent/Strategy/Layout 快照，重复 execute 只做 exact plan
type guard 与调用，不重新运行完整 graph/layout validator。

当前 c10d transport 在 C++ `execute()` 内对 ProcessGroup Work 调用 `wait()`；因此
transport 阶段仍会阻塞调用线程。`CudaWork` 只覆盖 collective 完成后的 CUDA kernel/event
完成语义，不能称为完整异步通信/计算 overlap。后续必须把 c10d Work completion、CUDA
stream/event 和 workspace ownership 串成非阻塞链。

### 8.1 A6000 性能证据与策略边界

证据环境为单机 NVIDIA RTX A6000，容器
`ccdl-comm-a6000:cu126-torch25`，PyTorch
`2.5.0a0+872d972e41.nv24.08`，CUDA 12.6；2 rank 使用物理 GPU 1、2，4 rank 使用
GPU 1、2、3、4。口径为 FP16 SUM、5 次 warmup、20 次计时迭代、每点 3 个独立 run；
每轮取最慢 rank CUDA event 延迟，每点再取 run-level 中位数。收益基线是相同进程组和
输入上的 PyTorch `dist.all_reduce`。

结果显示 2 rank 的 INT8 crossover 位于 4 MiB 与 16 MiB 之间：16 MiB、64 MiB 分别比
PyTorch native 快 28.91%、42.00%；256 KiB、1 MiB、4 MiB 分别回退 46.59%、
37.13%、3.21%。4 rank 的 256 KiB 至 64 MiB 全部回退 12.58%–42.74%。INT8
相对 L2 误差三轮中位数范围为 0.0136%–0.0900%，cosine 最低
0.999999762。

该结果的结构性原因是当前 compressed collective 为全量 all-gather：每个 rank 接收
`world_size` 份 payload。2 rank 大桶的字节压缩足以覆盖量化和 kernel 开销；4 rank 时
all-gather 扩展成本超过压缩收益。故当前策略矩阵只允许 2 rank、逻辑桶至少 16 MiB
进入 INT8 试用，其余已测组合选择 Native。要扩展到 4/8 rank 和多机，应增加真正的
compressed reduce-scatter、分层 collective 或 ReducedShard consumer，而不是继续扩大
全量 all-gather。

这些数字是通信源语证据，不是端到端训练吞吐或收敛证据，不能用于 Production-Auto
晋级。

### 8.2 CUDA 进程 placement

`lowbit_comm.backends.cuda.placement` 是 torch-free 的 ProcessGroup 前置边界。
`CudaProcessPlacement` 保存不可变的 rank→CPU tuple 和可选 NCCL channel；
`parse_cuda_process_placement()` 负责把 CLI 文本转为 canonical 值，
`apply_cuda_process_placement()` fresh 重建并验证完整配置后才提交进程状态。

```text
CLI / launcher values
-> parse exact CPU ranges and NCCL channel
-> immutable CudaProcessPlacement
-> validate rank/world, platform APIs, available CPUs and environment
-> sched_setaffinity + NCCL_MIN/MAX_NCHANNELS
-> initialize c10d ProcessGroup
```

placement 不进入 Strategy、Evidence key、plan cache 或 execute 热路径。空配置不读取或
修改 affinity；只有 channel 的配置不依赖 Linux affinity API。非空 CPU 映射需要
`sched_getaffinity`/`sched_setaffinity`，所有 CPU 必须属于调用进程原有 allowed set，
且不同 local rank 的映射不得重叠。NCCL min/max 任一已有值与请求不一致时 fail closed，
避免 launcher 环境被静默覆盖。

### 8.3 RSAG/qWD 与 CAG 发布边界

历史 PSI 证据覆盖单机 A6000 2/4 rank、三个 seed、三个 epoch，但当前硬化版不把历史
报告直接编译进 wheel。新 `RSAGQWDAdapter` 的选择流为：

```text
RSAGEnvironment + exact tuple[RSAGEvidence]
-> reject unknown / unsupported binary or runtime matrix
-> hash every installed lowbit_comm Python runtime file and the loaded _C binary
-> exact match world/node/message/topology/transport/GPU/Torch/CUDA/NCCL/version/ABI
   / checkpoint schema / build fingerprint
-> require one and only one record
-> require quality_passed and every seed speedup > 0
-> RouteDecision(rsag_qwd) or RouteDecision(native)
-> gather live rank/node/GPU/runtime identity again
-> qualified ReducedShard + private qWD ABI plans
```

空证据天然得到 Native。显式 `requested="rsag_qwd"` 在任一资格门失败时抛
`CapabilityError`；不会把 Native 静默报告为压缩成功。plan adapter 只在资格和 live
identity 通过后调用 extension 私有 `_create_qwd_plan`；稳定顶层、Backend capability 和
Production-Auto 都看不到该工厂。ShardedAdamW checkpoint schema v2 同时保存 exact shard
layout/rank 与 `force_refresh`，加载时先完整验证再原子提交。恢复复用 checkpoint 中的
cadence，不额外注入 refresh；schema v1 和跨 rank/layout 状态 fail closed。

RSAG evidence schema v2 的 `build_fingerprint` 不依赖 Git checkout 或安装绝对路径。检测器
递归读取安装包内全部 `lowbit_comm/**/*.py` 和实际加载的 `lowbit_comm._C` 文件，以相对
安装包根目录的逻辑名称排序，将逻辑名称、长度和内容分隔后计算 SHA256。任一文件缺失或
不可读均返回 capability failure；adapter 创建 plan 前重新检测并要求与证据环境完全相等。

`experimental.evidence` 提供唯一的外部 JSON 入口
`load_rsag_evidence_manifest(path, expected_sha256=...)`。入口先以无跟随方式固定普通文件、
大小和设备/inode/时间身份，读取后再次核对身份并校验调用方 pin；JSON decoder 拒绝重复
key 与非有限常量，随后按 exact schema 构造 frozen manifest/record。它没有默认路径、环境
变量或网络分支，也不把证据写入 wheel。loader 只解决来源固定和结构完整性，route selector
仍以 live `RSAGEnvironment` 执行一条且仅一条的精确匹配。

训练质量审计与 checkpoint 使用两个所有权边界：`state_dict()` 克隆 tensor 并用于持久化；
私有 `_audit_state()` 只在同步 step 边界借用当前 optimizer/EF tensor，立即流式复制到 CPU
计算 SHA256，随后丢弃。这样哈希字段与 checkpoint 等价，但不在 GPU 上先克隆同一份状态。

CAG capability 继续只支持 Explicit 诊断。CAG 训练：BLOCKED；其质量负证据不能驱动
experimental adapter 或 Production-Auto。当前 adapter 仅接受所有 seed 收益严格大于 0
的证据；这条 opt-in 规则不修改 Compiler schema-v3 的正式 Production-Auto promotion
门槛。

### 8.4 二进制兼容矩阵与异步边界

`experimental.compatibility` 使用 frozen `RSAGRuntimeABI` 和 exact tuple 保存已验证矩阵。
当前矩阵只有 Torch `2.5.0a0+872d972e41.nv24.08`、CUDA 12.6、NCCL 2.22.3、
扩展 ABI 1；GPU、
world size、node、topology 和 transport 继续由 `RSAGEnvironment`/Evidence 精确限定。
历史端到端测试记录使用运行源码
`0847d07e232703db104033582008a818f35d5443` 及其安装态构建指纹
`e50bd95f3de57c3458791bed0e4c4431f0866a3c6cb46ec3b6d73ca35da95a66`、
89,912,620 bytes 逻辑通信量、两节点 A6000 和 NCCL Socket/eno2。D2-NIC 与 D4-NIC
曾报告通过 3 seed/3 epoch、三种正收益口径、质量同源检查及 RSAG 精确恢复 oracle；
单机同工作负载不在资格范围内。

该指纹的 D2/D4 external wall 中位收益分别为 +23.60%/+3.95%，最小收益分别为
+23.27%/+3.76%；外部 evidence manifest 已用当前 selector 验证，仍不编译进 wheel。
上一 `52224b1a…` 指纹证据不得复用。外部 PSI workspace 状态使用 metadata object
collective 加逐 Tensor broadcast，Tensor payload 不再通过对象序列化。训练镜像对实际
执行的 Apex autocast helper 使用公开 AMP API，但没有把第三方 Apex 全树纳入本包稳定
接口或完整清理声明。

加入严格 loader 的候选源码 `378a7382bc28ab9ce55fe93bd6db1227bbb83a78` 使用可复现的
stripped 扩展 SHA-256 `21a1477c9f89cccb0ad97738b5aab49520d38c8b6dc6fd0d570ad7b4ace8b41b`，
安装态构建指纹为 `724dd75753d532e0d2be24ed3f8b7a6373e18a489e8c1d5062ac5aa3cb50ce7d`；两节点兼容性 smoke
已验证二进制一致、ABI=1 且未加载 Apex。该指纹尚无独占环境 3 seed/3 epoch 证据，selector
会拒绝 `e50bd95f…` manifest 并回退 Native，直到同范围正式重新资格完成。
`probe_rsag_compatibility()` 延迟导入 torch 与 extension loader，返回结构化 report；缺 CUDA、
缺 NCCL、extension 不可用、ABI 漂移或矩阵外版本都不抛出虚假成功。

扩展 loader 先验证 ABI，adapter 随后验证 live runtime matrix 与 evidence identity。
`supports_async=True` 只表示 collective 完成后的 CUDA kernel/event 尾部可由 CudaWork
查询和等待；transport 仍同步等待，因此当前架构不声明 transport overlap 或完整通信/
计算重叠。

### PSI 训练诊断运行时

CPU 可测试的 `tests/benchmarks/psi_training_runtime.py` 提供 `RankBatchSampler`、
`FP32MasterWeights`、`BatchWaitTimer`、`PhaseTimer`、`TrainingLoopWindowObserver` 与
训练协议校验。PSI sampler 输出全局交错 batches，
worker 在创建 DataLoader 后显式按 rank 取整批，保留原 collate/generator；epoch 索引物化和
精确恢复均基于局部分片。Native/CAG 在 DDP warmup 后将原 optimizer 的 param groups 绑定到
FP32 master，保留 scheduler/参数组超参身份；更新前转换梯度并 FP32 clip，更新后发布 FP16
权重，overflow 时主权重和 Adam 状态不变，清理模型与 master 两侧梯度。

Native 默认不注册 hook，使用 25 MiB bucket 的标准 DDP reducer；diagnostic 模式保留同步
单桶并在 FP16 SUM 前预除 world size。RSAG 的全局 norm clipping 系数在设备端计算，避免
读回 CUDA 标量；候选 optimizer、EF 回滚和 qWD 事务快照仍保留，不以减少拷贝为由破坏原子性。

所有路线使用两次不执行 optimizer 更新的共同模型 warmup，并恢复模型 buffer、梯度、
训练模式和 RNG；恢复 oracle 只观测，不额外推进 qWD refresh。确定性预取由
`psi_prefetch.py` 按 seed/epoch/绝对局部位置派生采样 RNG，使用独立 loader generator；
恢复使用已消费位置而非预取游标。protocol 的 data_pipeline 绑定模式、workers、
prefetch_factor、seed 和种子派生算法。该模式不承诺支持未覆盖的私有 RNG 或有状态变换。

checkpoint 的 training protocol 绑定 rank/world/batch、模型/optimizer/master 精度、
Native reducer、计时、共同 warmup、观测 oracle 及数据路径身份。diagnostic/production
生成 protocol v3，window 生成 protocol v4；历史合法 v2 可读，但实际恢复仍在修改训练
状态前要求 checkpoint protocol 与当前 engine protocol 精确相同。这不改变公开 adapter
checkpoint v2。独立 Native FP32 参考保留 FP32 模型参数并使用 FP16 autocast，不是全
FP32 算术，也不与 FP16 模型三路比较混算。

三种计时模式的区别如下（CLI 默认仍为 diagnostic）：

| 模式 | 协议/结果/raw schema | 测量含义 |
| --- | --- | --- |
| diagnostic | v3 / v3 / v1 | CUDA event 同步分阶段诊断 |
| production | v3 / v3 / v1 | 步前/后同步的整步 wall；写入 update_s，阶段零值表示不可用 |
| window | v4 / v4 / v2 | 观测窗口边界同步的训练循环 wall，单步与 exclusive-core 指标为 null |

window 在取 batch 前开启窗口，在本次本地 warmup 记录边界、epoch 结束或中途停止时关闭；
无零长度窗口。窗口记录实际局部/全局范围、epoch、样本数与 wall，各 rank 验证覆盖一致。
非审计步暂存 detached loss，窗口结束同步后将实际数值回填，再输出 raw；审计步立即读取。
`commit_observed_step` 保持 loss → measured quality audit → resume/oracle → record
build/append → midpoint oracle → window observe/reopen 顺序；epoch 末 oracle 显式消费
该步返回的 audit facts，不重算或使用缺省 hash。失败不输出成功结果，也不在异常 finally
中添加 CUDA 同步。算法要求的 Work/collective/AMP 等待不因窗口观测而删除。

窗口 wall 包含取数、H2D/augmentation、更新、循环内审计和 checkpoint 处理；epoch 验证
及 epoch-final checkpoint 位于窗口之外，延迟 loss 的主机读取发生在窗口计时结束后。
分别输出 `steady_training_loop_window_s` / `steady_training_loop_samples_per_second`，
无稳态覆盖时为 null；单步延迟/分阶段和 exclusive-core 吞吐为 null。window 结果中的
data/core 分量为 null，以免与循环窗口重复累加；process wall 为直接观测，外部控制器
启动至退出 wall 仍是端到端配对指标。不能将窗口 wall 当作覆盖全部训练作业开销的总时间。

每 rank raw 日志及 `.protocol.json` 保存独立更新计数、采样摘要、数据等待和源码/结果
散列；旧模式 sidecar 不增加 window 字段。标准 reducer 梯度通信不单独计时，字节量是
route-specific 布局/控制估算而非完整实测 wire，不能将未覆盖的 Native 梯度计数当作零。
worker 使用私有计划，不证明公开 adapter 资格，固定 `qualification_eligible=false`。
窗口模式代码和 CPU 契约通过不等于 CUDA 状态一致或性能合格；必须另行完成同模型/数据/
精度/拓扑/更新预算的多 seed 配对及恢复、溢出、全状态/RNG 验证，保留明确计时身份差异。

上述历史训练脚本存在未分 rank 消费全局 batch 与优化器精度不一致问题，历史端到端数值不得
继续作为质量或加速资格。回放数据 train/val 独立性未证实，不能说明收敛；新资格需完整独立数据、
公开 adapter、标准用户训练基线和配对多 seed/多 epoch 外部 wall 实验。收益门槛保持严格正向，
不恢复已取消的 5% 要求。

## 9. Reference 数值 oracle

ReferenceBackend 对一个完整 rank-value tuple 做确定性归约。FullTensor 为每个 rank 创建
独立结果对象；ReducedShard 使用确定 offset 和 padding 拆分全局归约值。SUM/MEAN、
多维 numel、非整除 shape、rank 多于元素、零元素、非有限输入和归约溢出都有契约测试。

Reference 的三个入口复用 Intent/Strategy 所有者的 trusted exact graph validator，
不动态调用输入对象上可被覆盖的校验方法。`compile_group()` 在校验后显式重建
TensorSpec、ShapeFamily、CommunicationIntent 和 StrategySpec；plan 每次执行前又
fresh 重验自身 exact 类型及嵌套图。因此 caller 或公开 plan 的事后篡改不会
绕过编译契约。对象图违反稳定为 CompileError；rank、数值或未知 oracle 运行
异常稳定为 ExecutionError，ReferenceGroupPlan 将同一 error 对象放入 FailedWork。

Reference 只验证语义，不模拟 NCCL、dtype rounding、设备异步、压缩 wire 或性能，不能
作为训练 Backend 或性能证据来源。

## 10. 公开导出和导入安全

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

## 11. 架构与发布门禁

- 根 `lowbit_comm/` 和根 `csrc/` 是唯一活动源码位置；
- 包版本唯一来源为 `lowbit_comm._version.__version__`，项目元数据通过 setuptools dynamic
  attr 读取同一值；
- 所有稳定类型的不可变性和非法组合都有 unit contract；
- Compiler 的 Explicit/Auto/Native、cache 和 signature 行为确定；
- facade compile-once、execute direct delegation 和 failure identity 通过 contract；
- Reference 明确不可注册为生产 Backend；
- 顶层精确导出和 isolated safe import 通过；
- experimental wheel 在无 torch 环境可导入，兼容性探针与 RSAG plan 创建保持延迟加载；
- 完整 pytest、Ruff、compileall 和仓库文档治理通过；
- `docs/` 只包含两份正式总文档，过程计划和任务报告不进入提交。
