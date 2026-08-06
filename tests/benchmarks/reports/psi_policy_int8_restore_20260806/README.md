# PSI Policy INT8 ReducedShard Restore 验证

## 结论

在 compressed reduce-scatter 之后继续量化 ReducedShard，以 INT8 完成
all-gather，并在 AdamW 使用梯度前反量化为 FP16，可以在当前 21 GB
真实数据训练上取得正收益。

| 配置 | 吞吐 | 相对旧 FP16 restore | 相对 Native DDP | P50 step | P95 step | epoch loss | val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 GPU INT8 restore | 199.183 samples/s | +7.10% | +6.94% | 159.070 ms | 176.110 ms | 3.881366 | 2.047609 |
| 4 GPU INT8 restore | 364.307 samples/s | +6.80% | +3.27% | 172.016 ms | 206.592 ms | 4.187568 | 2.945411 |

所有吞吐统计均丢弃前 20 steps。旧 FP16 restore 和 Native DDP 数值来自
同日同模型、数据、batch 和 epoch 的报告。4 卡新旧测试均严格使用 GPU
1、2、3、4；2 卡新测试使用 GPU 1、2，而历史基线使用 GPU 0、1，因此
2 卡百分比仍应通过后续重复运行确认。

## 实现路径

```text
compressed reduce-scatter
  -> FP ReducedShard
  -> INT8 quantize ReducedShard
  -> uint8 all-gather
  -> dequantize into final FP16 bucket slices
  -> standard DDP + AdamW
```

标准 `restore_mode="fp16"` 保持默认和 fallback。新路径通过
`restore_mode="compressed"` 显式启用，不改变 `CompressionConfig` 的算法职责。

## 微基准与数值正确性

4,194,304 元素、20 warmup、100 iterations：

| 配置 | FP16 restore pipeline | INT8 restore pipeline | pipeline speedup | restore 压缩率 |
|---|---:|---:|---:|---:|
| 2 GPU | 0.7371 ms | 0.7038 ms | 1.047x | 1.939x |
| 4 GPU | 1.5027 ms | 1.1675 ms | 1.287x | 1.939x |

- 两组 rank 间最大输出差异均为 0。
- 相对精确 FP all-reduce 的 relative L2 约 0.84%。
- 第二次 restore 量化相对 FP16 restore 增加的 relative L2 约 0.59%。
- 完整训练 loss 有限，完成全量 validation 和 checkpoint。

单 epoch、单 seed 不能证明最终任务精度等价，但本次没有发现发散、NaN、
rank 不一致或收敛步数增加。验证损失相对旧 FP16 restore 分别下降约 2.9%
和 2.4%，方向上没有显示新增量化导致训练恶化。

## 测试中发现并闭环的问题

最初真实训练在第一步触发 `CUDA error: misaligned address`。最小化到 2-rank、
128 元素后稳定复现。根因是每个 64 元素 FP16 INT8 group 的 packed payload
为 66 bytes，而生成的 CUDA decoder 使用 `int4` 向量加载。直接把多个 rank
payload 连续拼接会让后续 rank 起点不满足 16-byte 对齐。

最终实现将每个 rank 的 payload transport stride 向上对齐到 16 bytes；padding
不参与解码。128 元素 tail-bucket oracle、2/4-rank 大张量 oracle 和完整训练均
已通过。16-byte padding 对大 bucket 的通信量影响可以忽略。

另有一个历史预编译 `.so` 的 fused-EF smoke residual 断言失败；基础 quant、
dequant 和本次 restore oracle 均通过。该失败来自旧二进制与当前 Python 快照的
融合残差语义不一致，不在本次 restore 数据路径内，已单独保留为环境告警。

## 后续性能重点

当前方案仍执行第二次 quantize 和逐 rank dequantize kernel。下一步优先级应为：

1. 把 `dequant-reduce-mean + restore requantize` 融为单个 CUDA kernel，避免中间 FP ReducedShard 往返显存。
2. 把 gathered payload workspace 和最终 FP output 纳入内置 pool，消除逐 bucket 分配。
3. 用一个 fused unpack/dequant kernel 处理所有 rank payload，代替 world-size 次 kernel launch。
4. 通过 CUDA stream/event 将 INT8 all-gather 和后续 bucket 反向计算重叠。

完整数值见 [summary.json](summary.json)，成功运行的控制台、逐 step JSON、Hydra
overrides 和训练日志位于 `raw/`。
