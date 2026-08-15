# lowbit_comm 0.4.0.dev0

`lowbit_comm` 正在建立面向分布式训练通信的编译式架构。当前版本是 v0.4.0
Phase 1 基础契约：它提供不可变通信意图、策略、结果、编译计划、完成语义和窄公开
facade，但尚未提供 CUDA/NCCL、量化通信或可用于训练加速的生产 Backend。

因此，当前版本不声明训练吞吐、端到端加速或生产可用性。仓库内 Reference 实现仅用于
确定性数值 oracle；它没有生产 Backend 的 capability/lowering 接口，也不会注册到
Registry 或接入 Compiler/facade。

## 公开语义 API

顶层 `lowbit_comm` 与 `lowbit_comm.api` 只导出相同的 26 个稳定语义类型、异常和
facade。典型调用分为编译一次、执行多次：

```python
import lowbit_comm

communicator = lowbit_comm.compile_communicator(
    intent,
    lowbit_comm.NativePolicy(),
    context=context,
    compiler=compiler,
)
work = communicator.execute(value)
result = work.wait()
```

`compiler` 必须在调用前完成 Backend 注册和证据配置。Phase 1 没有随包提供生产
Backend，以上接口用于验证架构契约和后续 Backend 集成边界，不构成可运行的 CUDA
训练示例。

三类 Policy 的语义如下：

- `ExplicitPolicy`：只编译指定 `StrategySpec`；能力不匹配时明确失败，绝不回退。
- `AutoPolicy`：只有精确匹配 Production-Auto 证据且满足约束时选择候选策略；否则在
  编译期固化同语义 Native 计划。
- `NativePolicy`：直接要求同语义 Native 计划，不参与证据选择。

两类输出语义互不等价：

- `FullTensorResult` 表示每个 rank 都获得完整聚合结果。
- `ReducedShardResult` 表示每个 rank 只获得自己拥有的分片，并携带 global shape、
  offset、valid length、padding 和 owner rank 元数据。它不会隐式补做 full
  all-gather。

`compile_communicator()` 只调用 Compiler 一次，并把不可变 `ExecutionPlan` 绑定到
`CompiledCommunicator`。稳态 `execute()` 仅调用已绑定 Backend plan；它不查询
Registry、不重新选择策略、不编译、不执行运行时 fallback，也不复制或检查 tensor。

`LowbitCommError` 是稳定公开异常基类；`CompileError`、`CapabilityError` 和
`ExecutionError` 分别覆盖编译契约、能力选择和执行失败。它们可从 `lowbit_comm` 或
`lowbit_comm.api` 捕获，且两处导出与 `lowbit_comm.core.errors` 中的类保持对象身份一致。

## Phase 1 边界

已经交付：

- 不可变且强校验的 Intent、Policy、Strategy、Result 和 CompilationContext；
- capability 驱动且顺序确定的 Registry；
- 精确环境证据、Promotion gate、保守 Auto fallback 和计划缓存；
- 不可变 ExecutionPlan、compile-once/run-many facade；
- Completed/Failed Work 与 caller-driven error-feedback 事务状态机；该状态机保证候选
  residual 在 commit 前不可见、状态转换合法，并在 abort 时保留旧值和原始失败原因；
- 覆盖 SUM/MEAN、FullTensor、uneven ReducedShard 和 padding 的 Reference oracle。

尚未交付：

- CUDA 扩展、NCCL 集成、量化 Kernel 和生产 Backend；
- DDP、FSDP/分片训练 Adapter；
- INT8/INT4 自动策略、生产性能证据或训练加速保证；
- Phase 2 的 launch/completion token、stale completion/event 拒绝、CUDA stream
  ordering，以及设备 event/workspace 生命周期绑定。Phase 1 的 commit 由调用方在通信
  成功后断言，不验证 Work、event 或 token 身份。

## 验证

CPU-only 环境可安全导入包；导入顶层 API 不会加载 PyTorch 或
`lowbit_comm._C`。本地验证：

```bash
python -m pytest -q
python -m compileall -q lowbit_comm tests
python -m ruff check lowbit_comm tests
```

稳定需求和架构说明见：

- `docs/SOFTWARE_REQUIREMENTS_ZH.md`
- `docs/SOFTWARE_DESIGN_ZH.md`
