# lowbit_comm 0.4.0.dev0

`lowbit_comm` 正在建立面向分布式训练通信的编译式架构。v0.4.0 已完成 Phase 1
语义基础，并进入 Phase 2 CUDA 通信硬化：生产 `CudaBackend` 可绑定调用方显式提供的
c10d `ProcessGroup`，将 Native 或 INT8 FullTensor/ReducedShard 通信编译为可重复执行的
rank-local plan；受控 RSAG/qWD 集成位于可安装但非稳定的
`lowbit_comm.experimental` namespace。

当前 CUDA 路径已在单机 2/4 卡 NVIDIA RTX A6000 上验证 FP16/BF16、SUM/MEAN、
INT8 group size 16/32/64、尾部非整除张量、真实 workspace 复用和直接 base
all-gather。它仍不是通用训练产品：尚无通用 DDP/FSDP Adapter 或 8 卡/异构结论；
多机收益仅限下文精确绑定的 experimental RSAG/qWD 矩阵。仓库内 Reference 实现仍只
用于确定性数值 oracle，不会注册到生产 Registry。

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
- Compiler Evidence schema-v3 完整策略签名；schema-v1/v2 不再由运行时兼容读取；
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
质量审计必须通过，且所有 seed 收益严格大于 0。构建指纹按安装包内相对路径排序覆盖全部
`lowbit_comm/**/*.py` 与实际加载的 `_C` 二进制内容，不包含安装绝对路径。空证据、未知字段、
重复证据、任一 seed 非正收益、质量失败或
运行时身份漂移都回退 Native；显式强制不满足资格时抛出 `CapabilityError`。

当前代码不内置任何资格证据，所以新环境天然选择 Native。RSAG/qWD 仍是 opt-in
experimental 能力，不进入稳定顶层 API 或 `CudaBackend.capabilities()`。CAG 训练：BLOCKED；
它只保留诊断路径，不能获得 experimental RSAG 资格或 Production-Auto。

部署方可用 `lowbit_comm.experimental.load_rsag_evidence_manifest(path,
expected_sha256=...)` 加载仓库外 manifest。调用方必须同时提供显式路径和预先固定的
SHA-256；loader 不搜索默认目录、不读取环境变量或网络，并拒绝符号链接、非普通文件、
超限文件、读文件期间身份漂移、重复 JSON key、NaN/Infinity、未知/缺失字段和记录身份漂移。
加载成功只表示 manifest 结构与来源固定，仍须由 selector 对 live runtime 做精确匹配。

RSAG/qWD checkpoint 当前 schema 为 v2，精确绑定 shard layout/rank，并保存原始
`force_refresh` cadence。schema v1、跨 rank/layout 或类型不兼容状态会在修改 optimizer
前被拒绝；恢复不会无条件插入额外 FP refresh。

当前唯一经过真实 CUDA 构建和 adapter smoke 验证的二进制矩阵如下；新矩阵必须随同范围
多 seed/多 epoch 正收益与质量证据一起发布：

| GPU | Torch | CUDA | NCCL | 扩展 ABI | adapter smoke |
| --- | --- | --- | --- | ---: | --- |
| NVIDIA RTX A6000 | 2.5.0a0+872d972e41.nv24.08 | 12.6 | 2.22.3 | 1 | 2/4 rank |

历史测试的运行源码为 `0847d07e232703db104033582008a818f35d5443`，
安装态构建指纹为
`e50bd95f3de57c3458791bed0e4c4431f0866a3c6cb46ec3b6d73ca35da95a66`。
它曾在真实 PSI 数据上运行 3 seed、3 epoch 的 Native/RSAG 交替测试。逻辑通信量精确为
89,912,620 bytes，transport 为 NCCL Socket/eno2；下表收益均为相对 Native：

| 拓扑 | world/nodes | 核心吞吐中位收益 | worker wall 中位收益 | 外部 wall 中位收益 | 最小外部收益 | 通信字节减少 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D2-NIC | 2/2 | +64.19% | +24.28% | +23.60% | +23.27% | 20.93% | 历史诊断结果，待重新验证 |
| D4-NIC | 4/2 | +6.22% | +3.98% | +3.95% | +3.76% | 73.65% | 历史诊断结果，待重新验证 |

当时脚本报告两个拓扑的所有 seed 在三种性能口径上均严格为正，质量/同源检查和 seed
20260822 的 RSAG 精确恢复 oracle 均通过；任务在 `FutureWarning=error` 下运行且告警扫描为 0。
但该训练脚本与后续 4090 回放共用未按 rank 分批、Native/CAG FP16 optimizer 状态的旧协议，
因此这些数值不再作为有效训练加速或质量资格。表中的通信字节减少是脚本估算，不是实测线速流量。
外部 PSI 已删除内置 CCDL 通信实现，workspace 状态使用 Tensor collective；派生镜像只
替换实际执行的 Apex autocast helper，不宣称其他未执行的 Apex contrib 模块已完成清理。
包仍不内置证据；不得仅凭上述历史数值构造新的 `RSAGEvidence`。
单机 2-rank 的同工作负载外部 wall 存在负 seed，继续使用 Native。

上一正式构建 `6dcf4a2` / `52224b1a…` 的 D2/D4 external wall 中位收益为
+23.56%/+3.13%，只对其自身指纹有效。旧证据不得改写 fingerprint 后用于当前构建；
历史外部 evidence manifest 曾用 selector 验证 D2/D4 可选 RSAG；这只验证选择器身份匹配，
不能替代经修正训练协议重新取得的质量与性能资格。指纹或拓扑漂移时回退 Native。

严格 manifest loader 加入后的候选源码为
`378a7382bc28ab9ce55fe93bd6db1227bbb83a78`，安装态构建指纹为
`724dd75753d532e0d2be24ed3f8b7a6373e18a489e8c1d5062ac5aa3cb50ce7d`。该指纹已通过两节点
A6000 二进制/ABI/运行时 smoke，但尚未完成独占环境下的 3 seed/3 epoch 正式重新资格；
`e50bd95f…` manifest 对它是 stale evidence，当前必须回退 Native。

### PSI 训练诊断协议

`tests/benchmarks/distributed_psi_v040_worker.py` 使用按完整 batch 分发的 rank-local sampler；
训练、验证和恢复共用该规则，不能整除全局 batch 的采样配置直接拒绝。三路模型参数均为 FP16，
Native/CAG 和 RSAG 均使用 FP32 master/Adam 状态。Native 默认保留标准异步 DDP reducer
（25 MiB bucket），`--native-ddp-mode diagnostic` 才启用同步单桶 hook；后者在 FP16 SUM
之前预除 world size。标准模式的梯度通信耗时未单独测量，不能把仅计入控制通信的数字当作全量耗时。

结果 schema v3 内含 execution protocol；旧 schema v2 只用于读取历史记录，不能混合作资格。
每个 rank 输出自己的 raw JSONL，结果同目录的 `.protocol.json` 记录迭代数、成功更新数、
跳过数、数据等待时间和文件散列。迭代/optimizer step 都是每 rank 的计数，不除以 world size。
新 execution protocol v3 还绑定公共模型预热、只读 oracle、计时和数据流水线配置；历史
protocol v2 仅可读取，不能作为新运行的恢复状态。Checkpoint 绑定 rank、world size、batch、精度和 reducer 模式；旧训练 checkpoint 拒绝恢复，
这与公开 RSAG adapter 的 checkpoint schema 是不同协议。

`--timing-mode diagnostic` 保留逐阶段 CUDA 同步；`--timing-mode production` 只在完整训练步
边界同步，记录 CPU 调度和 CUDA 完成的 wall 时间。后者将完整 core 时间存入 `update_s`，
其余阶段为未测量的零占位，protocol 明确标记 `phase_breakdown_available=false`；不能把这些
零值解释为没有通信，稳态 core samples/s 也不能代替包含数据等待的端到端吞吐。

`--data-mode deterministic --loader-workers 2 --loader-prefetch-factor 2` 启用有界的多进程预取。
数据按 seed/rank/数据流、epoch 和实际采样位置确定 CPU 随机数；checkpoint 按已消费位置恢复，
不按预取队列位置恢复。设 workers=0 可作同数据语义对照，默认 legacy 模式保留历史同步读取。
Dataset 必须可 spawn 序列化、仅使用 CPU 和受控全局随机数，collate 必须确定性；私有 RNG、
有状态变换和 worker CUDA 操作不属于该契约。三路均做相同次数的模型预热并恢复 buffer/RNG；
恢复 oracle 仅观察，不再额外改变 RSAG 刷新策略。

本 worker 仍包含质量审计和私有 plan 调用，始终标记
`qualification_eligible=false`。标准 reducer 模式也是受控 FP16 模型对照，不等于通常的
FP32 参数 + AMP 用户训练基线。真实批次回放只能用于 smoke；正式收敛资格必须另外验证训练/
验证集独立性，并经公开 adapter、完整数据、多 seed/多 epoch 和外部 wall 测试取得。

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
