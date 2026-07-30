# CCDL 低比特高性能通信库软件需求规格说明书

## 1. 文档信息

- 文档状态：目标架构需求基线
- 版本：1.0
- 日期：2026-07-30
- 当前主要开发平台：NVIDIA GPU、CUDA、NCCL
- 当前主要验证环境：2/4 卡 NVIDIA RTX A6000
- 后续扩展目标：8 卡、单机多卡、多机多卡、Ascend及其他后端

## 2. 项目目标

CCDL是独立的低比特高性能通信库，为分布式训练和通用张量通信提供可压缩、可异步、可扩展的集合通信与点对点通信能力。

CCDL的第一优先级是通信性能。在保证数值正确性、训练状态一致性和故障可诊断的前提下，降低通信字节数、kernel launch开销、临时显存分配和CPU同步开销，并提高通信与计算的重叠比例。

CCDL不负责模型切分、训练生命周期、优化器、数据加载或任务调度。调用方负责选择通信策略；CCDL负责验证、编译和高性能执行该策略。

## 3. 核心原则

1. GPU优先：当前新增功能首先针对NVIDIA GPU、CUDA和NCCL实现与优化。
2. 性能优先：公共抽象不得使稳态热路径相对直接后端调用产生可观性能回退。
3. 显式策略：调用方显式指定通信策略；只有指定`auto`时才允许自动选择。
4. 严格执行：显式策略不受支持且未配置fallback时必须报错，不得静默切换。
5. 控制面与数据面分离：策略解析、能力验证和资源规划发生在初始化或cache miss阶段。
6. 后端隔离：公共协议统一，CUDA、Ascend和CPU实现、构建工具链及二进制产物相互隔离。
7. 单仓库多包：后端不得通过长期Git分支维护，应通过独立源码目录和构建产物隔离。
8. 安全异步：通信句柄、CUDA stream/event和workspace生命周期必须严格有序。
9. 可解释：每次执行可查询请求策略、实际策略、fallback、通信字节数和关键参数。
10. 可验证：所有性能结论必须建立在同口径真机benchmark和正确性检查之上。

## 4. 术语

- Communication Plan：调用方声明的完整通信计划。
- Communication Stage：分层通信中的一个执行阶段。
- Compiled Plan：完成能力验证、后端解析、资源规划后的不可变执行计划。
- Executor：Compiled Plan对应的后端热路径执行器。
- Strategy：all-gather-reduce、ring、tree、reduce-scatter等通信策略。
- Backend：CUDA/NCCL、Ascend/HCCL、CPU/Gloo等执行后端。
- ReducedShard：只包含本rank归约结果分片及其布局元数据的对象。
- Work：异步执行句柄，支持等待、查询和后端Future访问。
- Error Feedback：低比特量化误差残差反馈策略。

## 5. 范围

### 5.1 当前范围

- NVIDIA GPU上的低比特量化通信。
- CUDA扩展、NCCL通信、CUDA stream/event完成语义。
- 单机2/4卡性能开发和回归。
- 兼容任意world size的通用策略和元数据设计。
- 为8卡和多机多卡预留process group及拓扑能力。
- 保留Ascend代码，但不与当前GPU性能目标争抢优先级。

### 5.2 非当前范围

- 训练框架调度。
- 模型、优化器及checkpoint管理。
- 完整替代NCCL。
- 在首阶段为所有后端提供完全相同的性能特性。
- 通过长期Git分支维护CUDA、Ascend或CPU产品版本。

## 6. 功能需求

### FR-001 通信计划

CCDL必须提供不可变的`CommunicationPlan`，至少描述：

- collective类型；
- strategy；
- backend；
- 量化配置；
- 同步或异步模式；
- process group；
- fallback链；
- 输出布局；
- workspace策略；
- error feedback策略。

简单场景必须保留快捷API，复杂场景使用显式Communication Plan。

### FR-002 分层通信阶段

CCDL必须提供`CommunicationStage`，允许分别配置：

- 节点内策略和backend；
- 节点间策略和backend；
- 各阶段bit和group size；
- 各阶段process group；
- 各阶段是否压缩；
- 最终返回完整张量或ReducedShard。

### FR-003 计划编译

CCDL必须提供`compile(plan, context)`接口，返回可复用的Compiled Plan或Executor。

编译阶段必须完成：

- backend解析；
- capability验证；
- 拓扑和process group验证；
- shape/dtype/layout检查；
- kernel配置；
- chunk规划；
- workspace规划；
- fallback解析；
-执行元数据初始化。

### FR-004 策略注册

CCDL必须提供collective、strategy、backend三维注册机制。新增实现不得要求修改所有公共collective入口的条件分支。

### FR-005 显式策略语义

- 显式策略支持时必须按请求执行。
- 显式策略不支持且无fallback时必须抛出`UnsupportedCollective`。
- 只有显式配置fallback时才允许回退。
- 只有`strategy="auto"`时才允许CCDL自主选择策略。

### FR-006 集合通信

目标公共协议必须覆盖：

- all-reduce；
- all-gather；
- reduce-scatter；
- all-to-all；
- broadcast；
- reduce；
- gather；
- scatter；
- barrier。

首期GPU性能实现优先级为：

1. all-reduce；
2. reduce-scatter；
3. all-gather；
4. all-to-all；
5. 其他collective。

### FR-007 点对点通信

必须支持：

- send/recv；
- isend/irecv；
- 动态shape；
- 量化metadata传输；
- Work生命周期持有；
- tag和process group。

### FR-008 GPU通信策略

CUDA后端至少支持：

- all-gather-reduce；
- compressed all-reduce；
- compressed reduce-scatter；
- ring；
- tree；
- p2p；
- overlap-gather；
- overlap-p2p；
- overlap-tree；
- overlap-scale；
- hierarchical；
- sharded输出。

### FR-009 低比特量化

GPU后端必须支持：

- FP16/BF16/FP32输入；
- INT8；
- INT4；
- 分组线性量化；
- compact payload；
-动态shape；
- quantizer metadata；
- 可选随机量化；
- 可选Top-K误差补偿；
- Error Feedback。

INT8为首要生产路径，INT4在正确性和性能达标后启用。

### FR-010 融合kernel

GPU生产快路径应提供：

- fused quant-pack；
- fused multi-payload dequant-reduce；
- fused mean；
- fused Error Feedback更新；
- 可选写入ReducedShard；
- 无restored中间张量的执行方式。

不满足kernel约束时必须进入显式、可记录的安全路径。

### FR-011 异步Work

Work至少支持：

- `wait()`；
- `query()`；
- `get_future()`；
- 最终结果；
- 执行信息；
- 异常传播；
- 持有in-flight资源。

`query()`不得触发CPU同步或延迟计算。

### FR-012 Workspace

CCDL必须内置可选workspace cache/pool，按以下维度复用：

- backend；
- collective；
- strategy；
- bucket shape；
- dtype；
- world size；
- bit；
- group size；
- chunk配置。

workspace必须区分send、recv、reduced和临时metadata，并保证stream安全。

### FR-013 ReducedShard

ReducedShard必须包含：

- shard索引；
- shard长度；
-原始shape和numel；
- padding信息；
- world size；
- reduce语义；
- dtype；
- layout；
- transport；
- 可扩展metadata。

### FR-014 Backend能力

Backend必须提供：

- capability描述；
- plan编译；
- executor创建；
- stream/event适配；
- allocator/workspace适配；
- 支持的collective和策略；
- 支持的dtype、bit、shape；
- 诊断信息。

### FR-015 执行信息

每个Compiled Plan和Work必须可提供：

- requested strategy；
- executed strategy；
- backend；
- fallback是否发生及原因；
-各阶段策略；
-原始字节数；
-压缩字节数；
-压缩率；
-workspace命中信息；
-异步能力；
-kernel快路径或fallback路径。

## 7. 性能需求

### PR-001 热路径管理开销

- 策略解析不得在每个训练step重复执行。
- 稳态执行不得进行字符串策略匹配。
- 稳态执行不得重新探测backend capability。
- 稳态执行不得创建process group。
- 稳态执行不得重新创建CUDA stream。
- 管理层相对直接Backend Executor调用的额外开销必须低于1%。

### PR-002 内存分配

- 稳态重复bucket执行应达到零显式workspace分配。
- Work必须持有所有in-flight buffer。
- CUDA allocator生命周期必须使用stream安全机制。
- cache必须有显存上限、淘汰策略和可观测统计。

### PR-003 CPU同步

库代码不得在生产异步路径中无条件调用：

- `torch.cuda.synchronize()`；
- CUDA event synchronize；
- blocking collective。

同步只能由显式同步API、benchmark边界或安全fallback触发。

### PR-004 Kernel launch

生产INT8快路径目标：

- quant-pack最多一次主kernel launch；
- dequant-reduce-mean-EF最多一次主kernel launch；
- 不创建每rank restored中间张量；
- Python不逐payload启动反量化。

### PR-005 GPU验证门槛

每个性能改动必须在A6000上至少完成：

- 2卡；
- 4卡；
- 1 MiB、16 MiB及更大通信bucket；
- FP16和BF16；
- INT8；
- 同步与异步；
- 与当前CCDL、原生PyTorch/NCCL比较。

任何默认快路径不得在代表性大bucket上低于修改前版本。若小bucket存在回退，应设置显式阈值。

### PR-006 扩展性

- all-gather策略必须明确记录O(world size)接收和显存成本。
- 4卡以上优先提供compressed reduce-scatter或sharded路径。
- 8卡和多机路径不得要求固定卡数。
- topology选择不得硬编码只支持2/4/8卡。

## 8. 正确性与精度需求

### CR-001 Rank一致性

所有rank必须对逻辑相同的归约结果执行更新。量化只作用于传输表示，不得使各rank模型状态无控制地分叉。

### CR-002 数值检查

每个策略必须对比FP32或FP16参考结果，记录：

- relative L2；
- max absolute error；
- RMSE；
- 非有限值。

### CR-003 训练验证

生产策略必须完成真实模型训练验证，至少记录：

- 吞吐；
- loss曲线；
-验证指标；
-达到目标指标的step数；
-达到目标指标的墙钟时间；
-峰值显存。

### CR-004 异步一致性

同步和异步路径必须在相同量化配置下产生等价结果，并验证重复`wait()`、异常传播和buffer生命周期。

## 9. 可靠性需求

- 扩展缺失时安全import。
- backend不可用时给出可诊断错误。
- 显式策略不得静默fallback。
- callback最多执行一次。
- 异步错误必须在Work中传播。
- 动态shape和非整除shape不得越界。
- workspace不得在通信完成前复用。
- 进程组配置错误必须在compile阶段发现。

## 10. 包与发布需求

目标构建产物：

- `ccdl-core`；
- `ccdl-cuda`；
- `ccdl-ascend`；
- 可选`ccdl-cpu`。

要求：

- 各backend独立构建和安装；
- CUDA包不依赖CANN；
- Ascend包不依赖CUDA；
- core不硬依赖torch；
- 后端版本声明兼容的Core ABI；
- 不通过长期Git分支发布不同后端。

首阶段允许保持单一wheel，但源码边界和Backend Protocol必须先稳定。

## 11. 测试与验收

### 11.1 测试层次

- Core纯Python单测；
- Backend conformance测试；
- CUDA kernel单测；
- 2/4卡distributed smoke；
- 2/4卡性能benchmark；
- 真实模型训练；
- 8卡和多机扩展验证；
- 安装、卸载和extension缺失测试。

### 11.2 版本验收条件

一个GPU性能版本只有满足以下条件才可发布：

1. 完整单测通过。
2. CUDA扩展构建成功。
3. 2/4卡正确性通过。
4. benchmark保存原始JSON和环境信息。
5. 默认策略无性能回退。
6. 显式策略和fallback语义可验证。
7. Work和workspace无已知生命周期缺陷。
8. 文档、API和执行信息与实际实现一致。

## 12. 当前实现与目标差距

当前重构版已经具备量化codec、CUDA/CANN扩展、安全import、P2P、collective、topology、reduce-scatter、hierarchical原型、统一Work和workspace基础。

尚需实现的目标能力包括：

- Communication Plan和Stage；
- Backend Protocol和Registry；
- Compiled Executor；
- 多包构建；
- 严格显式策略；
- 执行信息；
- 完整Backend对等协议；
- C++/CUDA热路径调度；
- 真正流水化compressed collective。
