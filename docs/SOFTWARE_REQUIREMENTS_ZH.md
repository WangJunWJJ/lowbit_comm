# lowbit_comm 0.4.0 软件需求规格说明书

## 1. 文档信息

- 产品版本：0.4.0（当前开发包版本为 0.4.0.dev0）
- 当前阶段：Phase 1 — 架构与语义基础
- 变更等级：BREAKING
- 日期：2026-08-15
- 状态：Phase 1 验收基线

## 2. 产品目标

lowbit_comm 的目标是提供编译一次、执行多次的分布式通信架构。调用方用强类型
`CommunicationIntent` 声明通信语义，用 Policy 声明策略选择方式；Compiler 在执行前
完成能力匹配、证据选择和 Backend lowering，并产生不可变 `ExecutionPlan`。稳态执行
只调用已绑定的 Backend plan。

Phase 1 只建立 CPU 可验证的语义、编译、运行时和 Reference oracle 契约。它不交付
CUDA/NCCL 或量化生产 Backend，不构成训练加速产品，也不承诺吞吐或收敛收益。

## 3. 范围与边界

### 3.1 Phase 1 必须交付

- 不可变、slotted、严格校验的 Intent、Policy、Strategy、Result 和 Context；
- `FullTensor` 与 `ReducedShard` 两种不可混淆的输出语义；
- capability 驱动、稳定排序且拒绝歧义的 Backend Registry；
- 版本化 Evidence、Promotion gate 和精确匹配的 Production-Auto 选择；
- 显式严格、Auto 保守回退、Native 直达的 Compiler；
- 不可变 ExecutionPlan 和 compile-once/run-many facade；
- Work 完成/失败身份及 error-feedback 的事务语义；
- 不依赖设备的全 rank Reference 数值 oracle；
- 不加载 PyTorch/CUDA 扩展的顶层安全导入；
- 只包含稳定语义类型的窄顶层 API。

### 3.2 Phase 1 明确不交付

- CUDA 扩展、NCCL/HCCL 集成和任何设备 Kernel；
- INT8/INT4 量化生产通信路径；
- 可注册的 Reference 生产 Backend；
- DDP、FSDP、Sharded 或优化器 Adapter；
- 训练吞吐、端到端加速和收敛收益声明；
- Phase 2 才需要的 stream/event、process group 和设备 workspace 绑定。

## 4. 功能需求

### FR-001 通信意图

`CommunicationIntent` 必须显式包含 TensorSpec、ShapeFamily、SUM/MEAN reduction、
FullTensor/ReducedShard 输出、同步/异步完成、world size 和 rank。非法 shape、类型或
rank domain 必须在对象构造时拒绝。

### FR-002 策略语义

- `NativePolicy` 强制编译同语义 Native 计划。
- `ExplicitPolicy` 严格使用指定 `StrategySpec`；不合法或缺少 capability 时编译失败，
  不得静默回退。
- `AutoPolicy` 只可选择满足 constraints 且具有精确 Production-Auto 证据的策略；任何
  证据维度不匹配都不得被选中。Compiler 必须按确定性顺序检查全部
  exact Production-Auto 记录；某条的 constraints、CompilationContext 或 exact
  capability 不合格时必须继续下一条，只有全部不合格才产生 Native fallback。
  只允许最终选中的 Backend 进入 lowering。

Strategy 必须正交描述 compression、collective、topology、accumulation、error feedback、
overlap 和 workspace 预算。Phase 1 声明这些类型不表示设备实现已经存在。

### FR-003 输出语义

`FullTensorResult` 表示每个 rank 可消费完整聚合值。`ReducedShardResult` 必须携带
`ReducedShardMetadata`，包括 global shape、offset、valid length、padded length 和
owner rank。ReducedShard 不得隐式转换为 FullTensor，也不得无条件执行 full
all-gather。
`ReducedShardResult` 构造时必须 fresh 重验 exact metadata 的全部所有权、shape
和范围不变量，拒绝构造后被篡改的 metadata；泛型 value 保持不受限制。

### FR-004 Registry 与 capability

Registry 只接受声明完整 capability 且实现 lowering 协议的生产 Backend。每条 capability
必须持有一个精确类型、完整且不可变的 `StrategySpec`；请求策略任一字段不同都不得成为
候选或进入 lowering。候选顺序必须由包含完整 strategy 签名的 capability key 决定而非
注册顺序；重复精确 key 必须失败。注册必须先无副作用地静态验证非空、精确字符串类型的
`backend_id`。每个已有 capability entry 的 `backend_id` 只能由最初成功注册的同一
Backend 对象身份拥有；不同对象复用该 ID 必须只用对象身份比较、在解析或调用其
`capabilities()`/`lower()` 之前稳定抛出 `CapabilityError`，不得调用用户 `__eq__`。
同一对象可在后续成功注册中追加非重叠 capability，每次成功注册只增长一次 generation；
空、畸形或冲突批次不得占用 ID。随后必须静态验证可调用的 `capabilities()` 和 `lower()`；
不得求值 property、动态
`__getattr__` 或用户自定义 descriptor。合法协议实现包括普通 method、精确内置
`staticmethod`/`classmethod`、instance-dict callable 和已初始化 slot callable。
`capabilities()` 只调用一次，且必须返回非空 tuple，其中每项都是重新校验通过的精确
`BackendCapability`。Registry 必须以显式字段构造保存完全独立的可信 capability 快照，
其中嵌套 `StrategySpec` 与所有容器值均不得复用 Backend 返回对象的实例；Backend 在注册
成功后篡改原始 capability 图不得改变已提交 key、候选、generation 或 lowering。
`lower()` 在注册阶段只验证而不执行。协议或返回值畸形统一抛出
`CompileError`，重复 capability 保持抛出 `CapabilityError`。ID 所有权与 capability
entry 必须在整批校验完成后一起原子建立；任何失败都不得部分写入
Registry 或改变 generation。Registry 必须把静态解析得到的 bound `lower` callable 与
每条 capability 原子绑定；Compiler 只能调用该已验证 callable，不得重新动态读取
`backend.lower`。诊断和公开的 Backend match 仍为 `(capability, backend)` 二元组，但
每次返回的 capability 必须是内部可信快照的全新独立投影，不得暴露内部 capability 或
callable；调用方即使通过 `object.__setattr__` 篡改投影，也不得污染后续查询、Compiler
选择或 cache。compiler-only lowering resolution 只能按外部 capability 的完整值 key
定位内部 entry，并返回另一份新投影与已绑定 callable。内部 entry key 与可信快照必须
始终一致，漂移时稳定 fail closed。内部 entry 必须使用无可设置实例字段的 exact
tuple-backed envelope 原子保存 capability、owner 和 bound `lower`；每次读取必须重验
exact envelope、snapshot key、owner ID 及 saved/static callable 绑定。Registry 只在编译期使用。
Registry 的 `candidates(intent, strategy)` 必须在读取或遍历任何内部 entry、调用能力匹配
逻辑之前，复用 Intent/Strategy 所有者的 trusted validator fresh 重验两个 exact 完整图；
任何嵌套伪造统一抛出 `CompileError`，不得泄漏 `TypeError`/`AttributeError`。
backend-facing `BackendCapability.supports(intent, strategy)` 同样是完整信任边界：它必须
fresh 重验 capability 自身及两个请求图，再进入不导出的纯匹配逻辑。

Reference oracle 不实现 `capabilities()` 或 `lower()`，不得注册到 Registry、接入
Compiler 或经 facade 执行。

### FR-005 Evidence 与 Auto

Evidence key 必须绑定环境、intent、strategy、节点、workload 和 bucket 区间等完整维度。
当前 schema-v2 的 intent 键必须精确包含 tensor dtype 和完整 shape、ShapeFamily
的 max-numel 与 alignment、reduction、output、completion 和 world size；只有本地
rank 为了让同一 collective 的所有 rank 生成同一个 key 而被明确排除。
schema-v2 的 `strategy` 维度只是兼容既有格式的部分 token，不得单独称为 exact strategy
签名；完整 strategy 语义由该 token 与 topology、bit width、group size、error feedback、
overlap、wire size 等现有维度共同覆盖。每个 `StrategySpec` dataclass 字段必须显式分类到
至少一个现有维度；未知、遗漏、空分类或未知维度必须在生成 key 前抛出 `CompileError`，
新增字段必须显式升级 schema，不得在 schema-v2 内静默漂移。
缺少任一当前 schema-v2 必需维度的 key 必须被拒绝，不得按 schema-v1
的较小维度集合读取。
当前 schema-v2 必须持久化通信收益、暴露通信收益以及端到端晋级需要的全部度量；声明
状态必须与纯函数从度量导出的状态完全相等。通信回归门、长测准入门必须先于质量、
收敛、端到端收益、最差运行、seed 数和跨 workload 复现门，后者不得绕过前者。
只有通过全部门禁的 schema-v2 证据才能晋升为 Production-Auto。schema-v1 证据只可
保留作历史诊断，不得参与 Auto；其 key 和 record 指纹不得被静默改写为 schema-v2。
Auto 不得使用近似、部分、过期或无法重新验证的匹配。

EvidenceStore 的 `records` 容器在每次使用时必须仍是 exact tuple，否则稳定抛出
`CompileError`。单条构造后伪造的 current record 继续按候选级 fail closed 丢弃，合法
legacy record 继续只用于诊断；这两类候选都不得阻断后续有效 current record，候选耗尽
时仍允许 Native fallback。current/legacy record 必须复用各自唯一的递归 validator
重验 key、完整 strategy、metrics 和状态，不得另写一套漂移逻辑。
每次 lookup、Auto 枚举和 Evidence generation 必须共用同一 current-record
normalization pipeline。该 pipeline 只对 fresh-valid exact schema-v2 record 分组，
并以已验证 EvidenceKey 的 exact 标量/tuple 值构造 canonical identity；任何重复
key 组必须整组排除，不得依赖 first/last 输入顺序。重复组不得遮蔽其他
唯一有效记录。Legacy 记录不进入 Auto 或 generation，但其独立序列化和指纹必须兼容。

### FR-006 Compiler 与 ExecutionPlan

Compiler 必须在 compile 阶段完成输入校验、策略解析、能力选择、lowering、证据指纹和
计划签名。ExecutionPlan 必须不可变，并绑定 intent、strategy、Backend 标识、Backend
plan、origin、signature 和可选 evidence fingerprint。相同编译签名可稳定复用 lowering
结果，但缓存值不得作为公开 `ExecutionPlan` 返回。Compiler 必须保存从未暴露的内部
immutable cache entry；每次 cache miss 和 hit 均返回新的 `ExecutionPlan` wrapper、完整
intent/strategy 快照和新的 BackendPlan adapter。调用方通过 `object.__setattr__` 篡改任一
公开 plan 字段、嵌套语义图或 adapter，不得改变 cache 或后续 compile。
每次 compile 必须在读取 Evidence generation、查询 cache、遍历 Auto evidence、查询
Registry 或调用 lower 之前，重新校验 caller 提供的完整不可变对象图：
`CommunicationIntent` 及其 `TensorSpec`/`ShapeFamily`、三种 Policy 及其
`StrategySpec`/`AutoConstraints`、`CompilationContext` 及其
`EnvironmentFingerprint`。每一层必须是 exact class，并通过可信 class 上的 invariant
函数递归重验；不得动态调用输入对象可覆盖的 validator。caller 图畸形统一抛出
`CompileError`，不得泄漏 `TypeError`/`AttributeError`，也不得被 Auto 候选的
`except CompileError` 吞掉并静默改成 Native fallback。失败不得命中或污染 cache，且
Evidence 遍历、Registry candidates 和 lower 调用次数必须为零。
所有手写 canonicalizer 必须在序列化前，以共享的 exact-type dataclass 字段完整性 guard
核对当前字段分类；字段新增、删除、重命名、遗漏或未知分类统一 fail closed。该 guard
不得进入编码 payload 或改变既有 cache key、计划签名和 v1/v2 evidence fingerprint。
Compiler 的 intent cache 编码必须包含本地 rank；只有 Evidence collective key 排除 rank。
输入图通过 fresh validation 后必须以显式字段构造 trusted intent、policy 和 context
快照；lowering 收到的 intent/strategy 与内部 cache 语义图也必须互不别名。cache hit
必须重新校验 exact 内部 entry、其 canonical cache key、完整语义图、policy/origin/evidence
关系、opaque BackendPlan 静态结构和运行期 identity，并重算计划 signature；任何内部污染
稳定抛出 `CompileError`，不得执行被替换的 BackendPlan。正常 hit 不得重复 Registry
候选选择或 lowering。Native 与 fallback 的 canonical strategy 必须每次构造新值，不能
暴露可污染的 module singleton。
cache entry 本体必须是 exact tuple-backed envelope，使 opaque plan、identity 和 saved
execute 执行锚不能被 `object.__setattr__` 原位改写；手工替换整个 cache value 仍必须
经过完整结构、语义和静态 callable identity 校验。

BackendPlan 可封装不可复制的设备资源和 Backend 状态。CCDL 不承诺对该 opaque 执行状态
做 `copy`/`deepcopy`；它属于已注册 Backend 信任域。Compiler 只在编译期无副作用地静态
绑定其 `execute` callable，并为每个公开 plan 创建新的轻量 adapter。adapter 的
`execute(value)` 只能单行直通保存的 callable，不得查询 cache/Registry 或执行 validation，
且必须原样传递 Backend 返回的 `CommunicationWork` 身份。
lowering 必须调用 Registry 在注册时保存的 bound callable；Backend 抛出的
`LowbitCommError` 必须保持原对象与精确子类，其他普通异常必须以原异常为 cause
规范化为 `CompileError`。

### FR-007 公开 facade

`compile_communicator(intent, policy, context=..., compiler=...)` 必须调用
`compiler.compile()` 恰好一次。它必须在返回前拒绝非正式 intent/policy/context、缺少
compile 方法的 compiler、非 ExecutionPlan 返回值，以及没有可调用 `execute()` 的
Backend plan。对结构化 Compiler 返回的 exact `ExecutionPlan` 必须重新运行既有 plan
完整对象图不变量校验，递归重验 plan intent、strategy、Backend plan 结构、Backend ID、
origin、signature 与 evidence fingerprint；即使伪造值与请求按值相等，只要 exact 类型
或嵌套不变量失效也必须拒绝。caller 图畸形必须在调用 structural compiler 前失败；
`ExecutionPlan` subclass 继续 fail closed。

`CompiledCommunicator` 必须 frozen 且 slotted，只持有一个不可变 ExecutionPlan。
`execute(value)` 只能返回 `plan.backend_plan.execute(value)` 的原始 Work。它不得执行
Registry lookup、重新选策略、重新编译、fallback、all-rank gather、Reference 特判、
tensor 校验或 tensor 拷贝。

### FR-008 Work 与失败传播

`CommunicationWork[T]` 提供 `is_completed()`、`wait()` 和 `result()`。CompletedWork
必须稳定发布一个值；FailedWork 必须在 wait/result 中传播同一个原始 ExecutionError。
facade 不得包装、替换或吞掉 Backend 返回的失败 Work。

### FR-009 Error Feedback 事务

Phase 1 的 error-feedback 更新是 caller-driven 的 prepare、commit/abort 状态机。候选
residual 在 commit 前必须不可见；commit 是调用方在通信成功后作出的断言，不验证
CommunicationWork、completion event 或 launch token 的关联身份。abort 必须继续发布
旧 residual，并使后续非法操作保留原始 ExecutionError 作为原因；重复操作和其他非法
状态转换必须失败。

将事务绑定到 launch/completion token、拒绝 stale completion/event，以及 CUDA stream
ordering 属于 Phase 2，不是 Phase 1 的运行时保证。

### FR-010 Reference oracle

Reference oracle 必须以确定性 Python 数值计算覆盖 SUM、MEAN、FullTensor、
ReducedShard、非整除分片、padding、空 shard 和非法输入。它只服务契约和数值测试，
不得冒充生产 Backend，也不得用于证明训练性能。

`compile_group()`、直接 oracle 执行和 `ReferenceGroupPlan.execute_group()` 必须在
访问 rank 数值前 fresh 重验 exact `CommunicationIntent`/`StrategySpec` 完整对象
图；编译后的 plan 必须持有与 caller 不别名的语义快照。输入/对象图契约
违反报 `CompileError`，rank/数值执行失败报 `ExecutionError`；plan 返回的
`FailedWork` 必须保留同一个 `ExecutionError` 对象。

### FR-011 公开 API 与安全导入

顶层和 `lowbit_comm.api` 的 `__all__` 必须是完全一致的 26 项精确集合，只包含稳定
intent/policy/result、必要枚举、CompilationContext、CommunicationWork、facade，以及
LowbitCommError、
CompileError、CapabilityError、ExecutionError 四个稳定异常。两处异常导出必须与
`lowbit_comm.core.errors` 中的类保持对象身份一致。不得导出 Compiler、ExecutionPlan、
Registry、EvidenceStore、Backend loader、ReferenceBackend、`_C` 或任何旧 API 符号。

`import lowbit_comm` 在没有 torch 和二进制扩展时必须成功，且不得尝试加载二者。

## 5. 非功能需求

### NFR-001 热路径

稳态 `execute()` 不得解析策略、查询 Registry/Evidence、探测 capability、创建资源、
重新编译、透明重试或执行无条件同步。Phase 1 通过直接 Backend plan 调用锁定该边界。

### NFR-002 类型与不可变性

公共数据类型必须使用精确类型验证，拒绝布尔值冒充整数和可变容器冒充不可变签名。
嵌套 frozen 对象也必须在每个信任边界递归重验；共享默认对象不得使一个被伪造的 Policy
污染后续新 Policy。计划、结果和策略必须可稳定比较、缓存或安全共享。

### NFR-003 依赖方向

语义 API 不得导入 torch、CUDA 扩展或训练 Adapter。Reference 不得反向成为生产
Backend。执行 facade 不得依赖 Registry、Evidence 或策略选择逻辑。

### NFR-004 发布洁净度

包版本只在项目元数据中定义一个来源。仓库 `docs/` 只跟踪本需求文档和架构设计文档；
计划、任务报告、审计和 benchmark 原始报告不得提交。Python 代码每行不超过 79 字符。

## 6. Phase 1 验收

- unit、Backend contract、architecture contract 和完整 pytest 全部通过；
- isolated safe import 在中文绝对路径和无 torch/extension 拦截下通过；
- public `__all__` 与批准的精确集合相等；
- Ruff、compileall 和文档治理检查通过；
- Reference oracle 与 production Backend 边界测试通过；
- 版本为 0.4.0.dev0，且没有第二个运行时版本常量；
- Git 暂存区不包含计划、报告、`.superpowers` 或越界文件。
