# lowbit_comm 0.3.0

`lowbit_comm` 是面向 GPU 分布式训练的独立低比特通信库。0.3.0 是一次
BREAKING 架构替换：公开接口统一为强类型通信程序、编译上下文、Backend 与
compile-once/run-many executable，不兼容旧 CCDL Python API，也不依赖 ParaScale。

## 设计边界

- Core 仅描述 operation、output、wire 与 algorithm，不导入 Torch/CUDA。
- Backend 在编译期完成能力校验、lowering、process-group/stream/workspace 绑定。
- 热路径不做 registry 查询、策略字符串解析、capability probe 或隐式 fallback。
- 显式算法严格执行；只有 `AutoAlgorithm` 可依据版本化实测证据选择策略。
- `ReducedShard` 直接交给 sharded consumer，不执行最终完整梯度 all-gather。
- 2 卡 FullTensor 优先评估 compressed all-gather；4 卡已验证应优先评估 compressed
  RSAG。实际选择必须由匹配硬件、拓扑和工作负载的 evidence 决定。
- Ring、Tree 当前仅为 schedule/原型。Hierarchical FullTensor 已提供显式 INT8
  executable，但只在有效多节点拓扑、group size 64、non-compact INT8 和 fan-in
  不超过 8 时开放；尚未取得匹配双机证据，因此不会由 `AutoAlgorithm` 自动选择。

## Typed API

```python
from lowbit_comm import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
    compile,
)
from lowbit_comm.backends.cuda import CudaBackend
from lowbit_comm.compiler import BackendRegistry

program = CommunicationProgram(
    operation=ReduceMean(),
    output=FullTensor(DataType.FP16),
    wire=QuantizedWire(bit=8, group_size=64, compact=False),
    algorithm=CompressedReduceScatterAllGather(),
)
context = CompileContext(
    rank=rank,
    world_size=world_size,
    shape=tuple(bucket.shape),
    dtype=DataType.FP16,
    device_type="cuda",
    device_architecture="sm86",
)
registry = BackendRegistry()
registry.register("cuda", CudaBackend())
executable = compile(
    program,
    context,
    bindings=RuntimeBindings(process_group=process_group),
    registry=registry,
)
output = executable.run(bucket).wait()
```

独立 P2P、动态量化 all-gather 与原生 collective facade 位于
`lowbit_comm.backends.cuda`；DDP 与 sharded/qWD 状态位于
`lowbit_comm.adapters`。它们均不进入 Core。

## 已验证的 A6000 策略边界

在约 3355 万参数的 FP16 residual MLP、每 rank batch 32、32 MiB DDP bucket、
单机 A6000 的三轮交替测试中：

| 卡数 | 策略 | 训练吞吐 | 相对 Native |
|---:|---|---:|---:|
| 2 | Native NCCL | 7279.39 samples/s | baseline |
| 2 | CompressedAllGather | 6971.37 samples/s | -4.23% |
| 4 | Native NCCL | 7682.97 samples/s | baseline |
| 4 | Compressed RSAG | 8861.55 samples/s | +15.34% |

因此该口径下 2 卡应使用 Native，4 卡可使用 compressed RSAG。上述结果不是跨模型、
跨拓扑的通用承诺；`AutoAlgorithm` 只有收到精确匹配设备、拓扑、软件指纹、shape、dtype、
wire 和物理 primitive 的 `BenchmarkEvidence` 时才会采用压缩，否则显式回退 Native。

可复现实验入口为 `examples/train_ddp.py`。CUDA Gradient EF 不执行逐 bucket 的
`isfinite().all().item()` 主机同步；混合精度训练的 overflow 检测由 AMP/优化器负责。

## 公开任务的 Time-to-Quality 示例

`examples/train_cifar10.py` 使用公开 CIFAR-10 和 ResNet-18，对比 Native
DDP、compressed all-gather 与 compressed reduce-scatter/all-gather。每次
进程组只运行一种模式，对比时其余参数必须相同：

```bash
torchrun --standalone --nproc-per-node=2 examples/train_cifar10.py \
  --mode native --data-root /data/cifar10 --output native.json
torchrun --standalone --nproc-per-node=2 examples/train_cifar10.py \
  --mode compressed_rs_ag --data-root /data/cifar10 --output rsag.json
```

结果包含环境指纹、逐 epoch 验证准确率和 loss、全局训练吞吐、达到目标
准确率的时间以及跨 rank 模型状态差异。只有候选模式达到等价验证质量时，
吞吐提升才可表述为训练加速。

## 构建与验证

```bash
python -m pip install build
python -m build --wheel
python -m pytest tests/contract -q
```

CUDA 扩展使用包内 `src/lowbit_comm/backends/cuda/csrc` 原生资产构建。CPU-only
环境可以安全导入包并运行 Core/Reference 测试；CUDA executable 会在编译期明确拒绝
缺失的 native capability。

正式的软件需求与软件设计见：

- `docs/SOFTWARE_REQUIREMENTS_ZH.md`
- `docs/SOFTWARE_DESIGN_ZH.md`
