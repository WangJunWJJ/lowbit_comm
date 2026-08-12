# Quantized FullTensor Contract 与统一通信接口设计

## 1. 状态与目标

- 状态：已获用户设计确认
- 日期：2026-08-12
- 目标仓库：`lowbit_comm`
- 目标版本：`0.3.0`（破坏性架构大重构）
- 重构分支：`codex/v0.3.0-major-refactor`
- 首要后端：CUDA/NCCL

本设计修复 FullTensor 压缩路径的数据流错误，并建立统一的编译式通信接口。
正常压缩路径必须在网络传输阶段保持量化格式，直到每个 rank 在本地恢复最终
FullTensor。FP16/FP32 通信仅作为显式算法、安全回退或低频参数刷新，不得由
“最终输出为 FP16/FP32”这一要求隐式触发。

本设计不兼容旧 Python API。现有 CUDA/C++ 量化、融合反量化、workspace 与
transport 实现可作为 Backend operation 复用，但旧控制面不得成为新接口的运行时
依赖。

本次不新建 Git 仓库。现有仓库继续保留历史、性能证据、Issue、PR、发布地址与底层
实现资产；`codex/v0.3.0-major-refactor` 作为 v2 架构的直接重构起点。代码版本发布为
`0.3.0`，并在版本元数据、CHANGELOG、迁移文档和发布说明中明确标注
`BREAKING: Major Architecture Refactor`。由于项目尚未达到 `1.0`，`0.3.0` 可以承载
破坏性变更，但不得将其描述为普通功能小版本。

## 2. 问题定义

当前 `make_torch_compressed_reduce_scatter_all_gather()` 使用
`restore_mode="fp16" | "compressed"` 同时表达：

1. 最终输出布局；
2. 网络传输格式；
3. ReducedShard 到 FullTensor 的恢复算法；
4. fallback 行为。

其中默认 `restore_mode="fp16"` 会在 compressed reduce-scatter 后，对 FP
ReducedShard 执行全精度 all-gather。该路径数值正确，但会抵消第二阶段通信压缩
收益，也使“完整恢复”被错误理解为“全精度重传”。

旧版 CCDL 的 ring/p2p 数据流在最终恢复前重新量化归约分片，再以量化 payload
执行 all-gather。新架构应继承这一通信语义，并用融合 Kernel、强类型契约和安全
资源所有权替代旧版逐 payload Python 循环。

## 3. 已确认的设计决策

### 3.1 三个正交维度

公共语义不得再使用 `restore_mode`。通信程序分别声明：

- `operation`：归约、收集或点对点操作的数学语义；
- `output`：调用方要求的最终逻辑结果；
- `wire`：网络上传输的物理格式；
- `algorithm`：实现该语义的 collective 拓扑与阶段组合。

概念接口如下：

```python
program = CommunicationProgram(
    operation=ReduceMean(),
    output=FullTensor(dtype=DataType.FP16),
    wire=QuantizedWire(bit=8, group_size=64),
    algorithm=CompressedReduceScatterAllGather(),
)

executable = lowbit_comm.compile(program, context)
work = executable.run(tensor, out=output)
result = work.wait()
```

`output=FullTensor(FP16)` 只规定最终逻辑结果，不能推导出 FP16 wire。

### 3.2 两级 IR

Semantic IR 描述目标，不含 process group、CUDA stream、workspace 或 NCCL 细节：

```text
ReduceMean(LocalTensor) -> FullTensor
wire = Quantized(INT8, group_size=64)
```

Backend lowering 生成具体执行步骤：

```text
QuantizeDestinationChunks
-> CompressedAllToAll
-> FusedDequantReduceMeanRequantize
-> QuantizedAllGather
-> GatheredDequantWriteback
```

编译阶段完成语义验证、能力验证、策略选择、融合选择、workspace 规划和运行时资源
绑定。执行热路径不得重新解析策略字符串或选择 fallback。

## 4. 数据流

### 4.1 Quantized FullTensor 快路径

对 world size 为 `W` 的归约，正常压缩路径为：

```text
每个 rank 的本地完整梯度
-> 按目标 rank 切成 W 个逻辑分片
-> 量化并打包各目标分片
-> compressed all-to-all / reduce-scatter transport
-> 每个 rank 收到属于本地分片的 W 个量化贡献
-> fused dequant + reduce + mean + requantize
-> 每个 rank 持有一个已全局归约的量化 shard
-> all-gather 量化 shard
-> 每个 rank 在本地执行一次 gathered-dequant-writeback
-> 每个 rank 得到相同的完整 FP16/BF16/FP32 Tensor
```

约束：

1. 两个跨 rank 数据阶段均传输量化 payload；
2. 中间不得先生成 FP ReducedShard 再执行第二阶段通信；
3. 最后一阶段每个 rank 各自启动一次恢复 Kernel，而非只有某个 rank 恢复；
4. gathered-dequant-writeback 只负责反量化和按逻辑 offset 写回，不再次归约；
5. padding 不能进入最终逻辑输出；
6. 所有 rank 的最终 FullTensor 必须在定义的量化容差内一致。

若 Backend 不支持融合 `dequant-reduce-mean-requantize`，可以使用语义等价的非融合
量化中间实现，但不得退化为 FP shard 通信而不产生显式 fallback 记录。

### 4.2 ReducedShard 快路径

ReducedShard 是不同的输出契约：

```text
本地完整梯度
-> compressed reduce-scatter
-> 本 rank fused dequant + reduce + mean
-> ReducedShard
-> sharded training adapter 直接消费
```

该路径不得调用最终 all-gather。通信 Core 不负责将 ReducedShard 隐式提升为
FullTensor。

### 4.3 显式全精度路径

全精度通信通过独立类型表示：

```python
CommunicationProgram(
    operation=ReduceMean(),
    output=FullTensor(dtype=DataType.FP16),
    wire=FullPrecisionWire(dtype=DataType.FP16),
    algorithm=NativeAllReduce(),
)
```

允许用途：

- 用户显式选择 Native NCCL；
- 编译阶段基于能力或实测证据回退；
- correctness oracle；
- qWD 的低频 FP 参数 refresh。

全精度 fallback 必须写入不可变 ExecutionInfo，包括来源算法、目标算法、原因和
wire 变化。运行中 collective 已提交后禁止透明切换算法。

## 5. 类型与验证规则

首期需要的稳定类型：

```text
CommunicationProgram
Operation: ReduceSum | ReduceMean | Gather | Send | Receive
OutputType: FullTensor | ReducedShard
WireFormat: QuantizedWire | FullPrecisionWire
Algorithm: NativeAllReduce | CompressedAllGather |
           CompressedReduceScatterAllGather | CompressedReduceScatter
```

Verifier 至少拒绝：

- `ReducedShard` 输出配合要求最终 full all-gather 的算法；
- `QuantizedWire` 缺少量化配置；
- 输出 dtype 被当作 wire dtype；
- 声称 QuantizedWire、实际 lowering 中存在未记录的 FP collective；
- FullTensor 路径缺少跨 rank 一致输出阶段；
- ReducedShard 路径存在隐式完整恢复；
- Backend 不支持所请求 EF 域、bit、dtype、world size 或动态布局。

显式策略采用严格语义：不支持则编译失败。`auto` 才允许在编译阶段选择已验证的
候选或 Native fallback。

## 6. Backend 与运行时职责

CUDA Backend 负责：

- 将 Semantic IR lowering 成 NCCL/P2P 与 CUDA Kernel 阶段；
- 验证扩展 ABI 和融合 Kernel 能力；
- 规划量化 send/recv/reduced/gathered/output workspace；
- 绑定 process group、CUDA stream 和 completion event；
- 暴露实际 wire format、通信字节和 fast path 证据。

Core 不导入 Torch、CUDA 或具体 transport。Backend 不反向导入 DDP Adapter 或高层
快捷 API。

统一 `Work` 的完成条件为：

```text
collective 完成
+ gathered-dequant-writeback 完成
+ 需要的 EF 状态更新完成
+ 输出 event 对 consumer stream 可见
```

CompiledExecutable 拥有 workspace pool。每次运行获取 lease，只有完成 event 就绪后
才可回池。caller-owned output 在调用方释放前不得进入自动复用池。

## 7. Training Adapter 边界

DDP Adapter 构造 FullTensor 程序，并保证每个 rank 获取相同完整平均梯度。梯度
Error Feedback 基于本地待发送值和本地量化重构值更新，不使用全局归约输出作为本地
residual 参考。

Sharded Adapter 构造 ReducedShard 程序并直接消费本地分片。qWD 是 Sharded
Training Adapter 中的参数同步算法，不是 Core collective strategy：

```text
local FP32 master shard update
-> parameter delta / master-model difference
-> INT8 all-gather
-> fused dequant-add into replicated model parameters
-> periodic or recovery-triggered FP refresh
```

FP refresh 是低频纠偏或恢复行为，不能与正常“完整恢复”使用同一个名字或默认路径。

## 8. 错误处理与可观测性

编译期错误必须包含请求的 operation、output、wire、algorithm、Backend 能力以及拒绝
原因。运行时不得在部分 collective 已提交后自动重试另一策略。

ExecutionInfo 至少记录：

- requested/effective algorithm；
- requested/effective wire；
- output contract；
- quantized bytes、full-precision bytes；
- fused stage 列表；
- workspace 预算与复用状态；
- fallback 原因；
- benchmark evidence ID。

因此调用方可以证明某次 FullTensor 的完整恢复是否全程使用量化 wire，而不是仅凭
策略名称推断。

## 9. TDD 与验证门禁

实现严格采用 Red-Green-Refactor。首批失败测试覆盖：

1. `FullTensor + QuantizedWire` 两个跨 rank 阶段都只接受量化 payload；
2. 默认压缩编译结果不得选择 FP shard all-gather；
3. gathered-dequant 每个 rank 只调用一次恢复 operation；
4. FullTensor 与全精度 reference 在配置容差内一致；
5. 所有 rank 的 FullTensor 一致；
6. ReducedShard 路径不调用最终 all-gather；
7. FullPrecisionWire 只能显式请求或产生可观察 fallback；
8. output dtype 与 wire format 正交；
9. workspace lease 在完成 event 前不得回收；
10. 旧 `restore_mode` 不出现在新公共接口。

分布式门禁：

- 单机 2 卡 A6000；
- 单机 4 卡 A6000；
- 双机 4 卡 A6000；
- 双机 8 卡 A6000。

性能采用 Native、旧压缩路径、新 Quantized FullTensor 路径交替至少三次，比较中位数
和离散度。既有生产快路径中位吞吐退化不得超过 2%。只有具有匹配硬件、拓扑、
world size、dtype、bucket、wire 和输出契约证据的策略才能进入 `auto`。

## 10. 迁移与删除

迁移按功能矩阵进行，但目标架构不保留双轨公共接口：

1. 重新规整 `0.3.0` 软件需求说明、架构设计说明和机器可检测架构契约；
2. 建立旧功能、旧性能与新版本验收要求的迁移矩阵；
3. 建立强类型 Program、类型和 verifier；
4. 建立统一 compile/run 接口；
5. 接入 Quantized ReducedShard executor；
6. 接入 Quantized FullTensor 两阶段 executor；
7. 将 DDP Adapter 切换到统一编译入口；
8. 将性能与正确性门禁全部跑通；
9. 删除新架构中的 `restore_mode` 和重复 transport 入口；
10. 必选功能 Ready 后删除旧控制面。

需求、架构与机器契约必须先于生产代码修改完成并单独提交。若后续实现需要改变 Core
ABI、IR、Backend Protocol、Work 完成语义或 workspace ownership，必须先修改并重新
审查相应文档，不能由实现自行选择解释。

旧实现可在迁移分支中作为数值 oracle 和性能基线使用；最终合并产物的 Native fallback
由新 Backend 实现，不调用旧 CCDL 控制面。

## 11. 非目标

- 不重写已经验证的 CUDA/C++ 量化 Kernel；
- 不建设任意图通用通信编译器或万能 Kernel 生成系统；
- 不承诺所有 workload 或 world size 都通过压缩获得加速；
- 不在 Core 中管理 optimizer、FP32 master weights、checkpoint 或 loss scaling；
- 不以 8 机 64 卡作为首版重构的阻塞门禁，但 IR、metadata 和 workspace key 不得写死
  2/4/8 卡。

## 12. 完成定义

本修复完成需要同时满足：

- 新接口能明确区分 output、wire 与 algorithm；
- Quantized FullTensor 正常路径不存在 FP 中间 collective；
- ReducedShard 路径不存在隐式完整恢复；
- FullTensor 数值、跨 rank 一致性和异步资源生命周期通过测试；
- A6000 2/4 卡完成同口径性能门禁，双机 4/8 卡完成正确性和规模验证；
- 新生产路径相对其旧生产基线退化不超过 2%；
- 新公共接口不包含 `restore_mode`；
- 旧控制面只在全部必选功能达到 Ready 后删除。
