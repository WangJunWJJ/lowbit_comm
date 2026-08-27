# lowbit_comm 0.4.0.dev0

`lowbit_comm` 正在建立面向分布式训练通信的编译式架构。v0.4.0 已完成 Phase 1
语义基础，并进入 Phase 2 CUDA 通信硬化：生产 `CudaBackend` 可绑定调用方显式提供的
c10d `ProcessGroup`，将 Native 或 INT8 FullTensor/ReducedShard 通信编译为可重复执行的
rank-local plan；受控 RSAG/qWD 集成位于可安装但非稳定的
`lowbit_comm.experimental` namespace。

当前 CUDA 路径已在单机 2/4 卡 NVIDIA RTX A6000 上验证 FP16/BF16、SUM/MEAN、
INT8 group size 16/32/64、尾部非整除张量、真实 workspace 复用和直接 base
all-gather。它仍不是通用训练产品：尚无通用 DDP/FSDP Adapter、8 卡/异构结论或经过
当前 schema-v2 重新资格化的多机收益声明。仓库内 Reference 实现仍只用于确定性数值
oracle，不会注册到生产 Registry。

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

`compiler` 必须在调用前完成 Backend 注册和证据配置。使用 CUDA 时，调用方还必须向
`CudaBackend` 显式传入已初始化的 c10d `ProcessGroup`；CCDL 不创建、替换或猜测默认
进程组。

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

## 当前交付边界

已经交付：

- 不可变且强校验的 Intent、Policy、Strategy、Result 和 CompilationContext；
- capability 驱动且顺序确定的 Registry；
- 精确环境证据、Promotion gate、保守 Auto fallback 和计划缓存；
- 不可变 ExecutionPlan、compile-once/run-many facade；
- Completed/Failed Work 与 caller-driven error-feedback 事务状态机；该状态机保证候选
  residual 在 commit 前不可见、状态转换合法，并在 abort 时保留旧值和原始失败原因；
- 覆盖 SUM/MEAN、FullTensor、uneven ReducedShard 和 padding 的 Reference oracle。
- 可安全探测/加载的 CUDA 扩展边界、精确 FullTensor CUDA capability/lowering、编译期
  workspace layout，以及基于原生 event 的 CudaWork、LaunchToken 和 workspace lease。
- 真实 c10d/NCCL rank-local FullTensor Native all-reduce；
- INT8 compact quantize-pack、量化 payload all-gather 和单 kernel
  dequant-reduce，支持 FP16/BF16、SUM/MEAN、2/4 rank 与 group size 16/32/64；
- CUDA FullTensor Native/INT8 均返回稳定 `FullTensorResult`，workspace 完成后按精确大小
  复用，INT8 FullTensor 使用直接 `_allgather_base`；
- wheel 内的 experimental RSAG/qWD 状态、版本化 checkpoint、证据路由和受控 plan
  adapter；顶层稳定 26 项 API 不变。

尚未交付：

- 通用 DDP、FSDP/分片训练 Adapter；
- INT4、INT8 Production-Auto 证据和跨拓扑端到端训练加速/收敛保证；
- 8 rank、多机、reduce-scatter/分层压缩 collective 的真机结论；
- error-feedback 与 launch token 的强绑定、stale completion/event 拒绝，以及跨 stream
  完整 ordering。当前 CudaWork 已绑定 native event/workspace 生命周期，但 Phase 1 的
  error-feedback commit 仍由调用方断言，不验证 Work、event 或 token 身份。

## Experimental RSAG/qWD 与 Native 回退

Native 默认。`lowbit_comm.experimental.RSAGQWDAdapter` 只有在一条证据同时精确匹配
world size、节点数、逻辑通信量区间、拓扑、transport、GPU、Torch/CUDA/NCCL、
`lowbit_comm` 版本、扩展 ABI、checkpoint schema 和安装态构建指纹时才选择 RSAG/qWD；
质量审计必须通过，且所有 seed 收益严格大于 0。构建指纹覆盖 RSAG 关键 Python 模块与
实际 `_C` 二进制内容。空证据、未知字段、重复证据、任一 seed 非正收益、质量失败或
运行时身份漂移都回退 Native；显式强制不满足资格时抛出 `CapabilityError`。

当前代码不内置任何资格证据，所以新环境天然选择 Native。RSAG/qWD 仍是 opt-in
experimental 能力，不进入稳定顶层 API 或 `CudaBackend.capabilities()`。CAG 训练：BLOCKED；
它只保留诊断路径，不能获得 experimental RSAG 资格或 Production-Auto。

RSAG/qWD checkpoint 当前 schema 为 v2，精确绑定 shard layout/rank，并保存原始
`force_refresh` cadence。schema v1、跨 rank/layout 或类型不兼容状态会在修改 optimizer
前被拒绝；恢复不会无条件插入额外 FP refresh。

当前唯一经过真实 CUDA 构建和 adapter smoke 验证的二进制矩阵如下；新矩阵必须随同范围
多 seed/多 epoch 正收益与质量证据一起发布：

| GPU | Torch | CUDA | NCCL | 扩展 ABI | adapter smoke |
| --- | --- | --- | --- | ---: | --- |
| NVIDIA RTX A6000 | 2.5.0a0+872d972e41.nv24.08 | 12.6 | 2.22.3 | 1 | 2/4 rank |

`probe_rsag_compatibility()` 可在加载 plan 前探测该矩阵。`CompletionMode.ASYNC` 和
Backend 的 `supports_async=True` 当前只表示 collective 后 CUDA event 尾部；
transport 仍同步等待，不能解释为通信/计算 overlap。

## 历史 A6000 FullTensor 通信证据

以下为优化前同一容器、同一 GPU 的历史微基准，用于说明通信量 crossover；当前
workspace/direct-all-gather 版本须以重新运行的同范围结果为准。口径为 FP16 SUM、
5 次 warmup + 20 次计时，每个点独立运行 3 次并取
run-level 中位数。延迟采用每轮最慢 rank 的 CUDA event 时间；收益为
`PyTorch native latency / CCDL INT8 latency - 1`。该指标是通信微基准，不是训练吞吐。

| ranks | 逻辑桶 | PyTorch native | CCDL Native | CCDL INT8 | INT8 收益 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 256 KiB | 0.136 ms | 0.196 ms | 0.255 ms | -46.59% |
| 2 | 1 MiB | 0.175 ms | 0.282 ms | 0.279 ms | -37.13% |
| 2 | 4 MiB | 0.463 ms | 0.532 ms | 0.478 ms | -3.21% |
| 2 | 16 MiB | 1.626 ms | 1.678 ms | 1.261 ms | +28.91% |
| 2 | 64 MiB | 6.331 ms | 6.258 ms | 4.458 ms | +42.00% |
| 4 | 256 KiB | 0.182 ms | 0.269 ms | 0.317 ms | -42.74% |
| 4 | 1 MiB | 0.292 ms | 0.398 ms | 0.471 ms | -38.04% |
| 4 | 4 MiB | 0.956 ms | 1.048 ms | 1.213 ms | -21.18% |
| 4 | 16 MiB | 3.577 ms | 3.781 ms | 4.212 ms | -15.07% |
| 4 | 64 MiB | 14.077 ms | 14.189 ms | 16.104 ms | -12.58% |

因此当前证据支持的显式建议是：2 rank 且单次逻辑通信量至少 16 MiB 时可试用 INT8
compressed all-gather-reduce；2 rank 小桶和全部已测 4 rank 桶必须回退 Native。
4 rank 回退的根因是 all-gather 接收 `world_size` 份 payload，其扩展性不及 NCCL
all-reduce；后续性能路径应改为 compressed reduce-scatter 或分层 collective。
INT8 三轮中位数的相对 L2 误差为 0.0136%–0.0900%，cosine 不低于
0.999999762；这只证明单次 collective 数值误差，不证明训练收敛。

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
