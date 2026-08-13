# 从 0.2.x 迁移到 lowbit_comm 0.3.0

0.3.0 是 `BREAKING: Major Architecture Refactor`。旧 `ccdl_comm` Python API、字符串
`CommunicationPlan`、`restore_mode`、ParaScale 专用 plugin 和直接 transport 工厂不再
属于公共 API。

旧方式把输出格式和 wire 混在一起：

```python
transport = make_torch_compressed_reduce_scatter_all_gather(
    restore_mode="compressed",
)
```

0.3.0 分别声明语义：

```python
program = CommunicationProgram(
    operation=ReduceMean(),
    output=FullTensor(dtype=DataType.FP16),
    wire=QuantizedWire(bit=8, group_size=64),
    algorithm=CompressedReduceScatterAllGather(),
)
executable = lowbit_comm.compile(program, context, bindings=bindings)
result = executable.run(tensor).wait()
```

DDP 使用 `lowbit_comm.adapters.ddp`；分片训练与 qWD 使用
`lowbit_comm.adapters.sharded`。框架集成通过 Adapter Protocol 实现，不依赖 Core 内置
plugin。

旧版本继续通过 Git tag/release 获取。0.3.0 不提供长期兼容包装，因为兼容入口会重新
引入多套策略权威、模糊输出/wire 语义和跨层依赖。
