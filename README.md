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
registry.register(CudaBackend())
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

## 构建与验证

```bash
python -m pip install build
python -m build --wheel
python -m pytest tests/v03 -q
```

CUDA 扩展使用包内 `src/lowbit_comm/backends/cuda/csrc` 原生资产构建。CPU-only
环境可以安全导入包并运行 Core/Reference 测试；CUDA executable 会在编译期明确拒绝
缺失的 native capability。

软件需求、架构契约、迁移状态与完整开发门禁分别见：

- `docs/SOFTWARE_REQUIREMENTS_ZH.md`
- `docs/ARCHITECTURE_BASELINE_ZH.md`
- `docs/MIGRATION_MATRIX_0.3.0_ZH.md`
- `docs/superpowers/plans/2026-08-12-v0.3.0-major-refactor.md`
