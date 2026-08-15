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
  证据维度不匹配都必须在编译期产生 Native fallback。

Strategy 必须正交描述 compression、collective、topology、accumulation、error feedback、
overlap 和 workspace 预算。Phase 1 声明这些类型不表示设备实现已经存在。

### FR-003 输出语义

`FullTensorResult` 表示每个 rank 可消费完整聚合值。`ReducedShardResult` 必须携带
`ReducedShardMetadata`，包括 global shape、offset、valid length、padded length 和
owner rank。ReducedShard 不得隐式转换为 FullTensor，也不得无条件执行 full
all-gather。

### FR-004 Registry 与 capability

Registry 只接受声明完整 capability 且实现 lowering 协议的生产 Backend。每条 capability
必须持有一个精确类型、完整且不可变的 `StrategySpec`；请求策略任一字段不同都不得成为
候选或进入 lowering。候选顺序必须由包含完整 strategy 签名的 capability key 决定而非
注册顺序；重复精确 key 必须失败。Registry 只在编译期使用。

Reference oracle 不实现 `capabilities()` 或 `lower()`，不得注册到 Registry、接入
Compiler 或经 facade 执行。

### FR-005 Evidence 与 Auto

Evidence key 必须绑定环境、intent、strategy、节点、workload 和 bucket 区间等完整维度。
当前 schema-v2 的 intent 键必须精确包含 tensor dtype 和完整 shape、ShapeFamily
的 max-numel 与 alignment、reduction、output、completion 和 world size；只有本地
rank 为了让同一 collective 的所有 rank 生成同一个 key 而被明确排除。
缺少任一当前 schema-v2 必需维度的 key 必须被拒绝，不得按 schema-v1
的较小维度集合读取。
当前 schema-v2 必须持久化通信收益、暴露通信收益以及端到端晋级需要的全部度量；声明
状态必须与纯函数从度量导出的状态完全相等。通信回归门、长测准入门必须先于质量、
收敛、端到端收益、最差运行、seed 数和跨 workload 复现门，后者不得绕过前者。
只有通过全部门禁的 schema-v2 证据才能晋升为 Production-Auto。schema-v1 证据只可
保留作历史诊断，不得参与 Auto；其 key 和 record 指纹不得被静默改写为 schema-v2。
Auto 不得使用近似、部分、过期或无法重新验证的匹配。

### FR-006 Compiler 与 ExecutionPlan

Compiler 必须在 compile 阶段完成输入校验、策略解析、能力选择、lowering、证据指纹和
计划签名。ExecutionPlan 必须不可变，并绑定 intent、strategy、Backend 标识、Backend
plan、origin、signature 和可选 evidence fingerprint。相同编译签名可稳定复用缓存。

### FR-007 公开 facade

`compile_communicator(intent, policy, context=..., compiler=...)` 必须调用
`compiler.compile()` 恰好一次。它必须在返回前拒绝非正式 intent/policy/context、缺少
compile 方法的 compiler、非 ExecutionPlan 返回值，以及没有可调用 `execute()` 的
Backend plan。

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
计划、结果和策略必须可稳定比较、缓存或安全共享。

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
