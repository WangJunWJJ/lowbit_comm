# CCDL 正确性、异步语义与 Kernel 加固设计

## 1. 目标

本设计修复重构版 CCDL 审计中发现的 P0、P1 问题，并在正确性门禁通过后继续优化 CUDA Kernel。实现必须同时满足：

- 未显式启用压缩时，DDP 行为与 PyTorch DDP 的归约语义一致；
- 压缩 collective 只能在 transport 明确支持其编码格式时启用；
- error feedback 使用可证明、可测试的本地量化误差；
- Future 完成表示 CCDL 的通信、解码、归约、反馈更新和 CUDA 有序完成全部结束；
- CUDA 入口在多设备、非当前设备和异步 stream 场景下安全；
- Kernel 优化不得以改变数学语义或降低已验证性能为代价。

## 2. 非目标

- 不兼容重构前 CCDL API；
- 不在本阶段为未知硬件自动开启压缩；
- 不重写 NCCL；
- 不让 full-output DDP 伪装成可直接消费 ReducedShard；
- 不在缺少动态证据时扩大自动策略表覆盖范围。
- 不把仅用于 pytest 的 benchmark 脚本当作用户示例；`examples/` 必须能由用户直接运行。

## 3. 方案选择

采用“能力门控的正确性重构”。与局部打补丁相比，该方案将数学语义、transport capability、异步完成条件和 fallback 统一到编译及执行契约中；与一次性重写 compressed collective 相比，它允许逐项 TDD、独立提交和性能回归定位。

## 4. 数学语义

### 4.1 DDP 归约

令 rank `i` 的本地梯度为 `g_i`，world size 为 `N`：

- `sum` 输出必须为 `sum(g_i)`；
- `mean` 输出必须为 `sum(g_i) / N`；
- 所有 rank 必须得到相同的 full-output 结果；
- ReducedShard 只要求每个 rank 得到全局归约结果的确定分片。

归约模式必须作为显式参数传入 executor，不能由 transport、Hook 或调用位置隐式推断。

### 4.2 压缩 payload

普通 CCDL payload 是包含量化值和 scale/metadata 的编码对象。Torch/NCCL 对其 `uint8` buffer 做逐字节 `SUM` 不具备数值等价性，因此：

- 默认 transport 不得对编码 payload 调用普通 `dist.all_reduce`；
- compressed-reduce transport 必须声明它理解对应 payload codec；
- capability 必须包含 codec、bit、group size、dtype、collective、output layout 和 async 支持；
- capability 不匹配时在编译期拒绝或 fallback 到 `native_nccl`；
- 不允许在运行期静默选择数学语义不同的路径。

### 4.3 Error Feedback

每个 rank 独立维护 residual。第 `t` 步：

```text
prepared_i = gradient_i + residual_i
payload_i = Q(prepared_i)
local_restored_i = D(payload_i)
residual_i = prepared_i - local_restored_i
global_result = Reduce(local_restored_0, ..., local_restored_N-1)
```

全局归约结果只用于模型更新，不用于计算本地 residual。residual 的更新频率仍由现有 policy 控制；跳过更新时必须保留既定 policy 语义，不得误清空 residual。

融合 Kernel 必须产生与上述分步参考实现等价的 `global_result` 和 `residual_i`。

## 5. 公共接口与能力门控

### 5.1 默认行为

`create_ddp_comm_hook()` 默认使用 `strategy="native_nccl"` 和 `reduce="mean"`。压缩策略只能由调用者显式指定，或由经过验证的 compile-time strategy table 选择。

旧的 `all_reduce` 名称不再自动绑定 `make_torch_all_reduce()`。如果调用者注入 transport，该 transport 必须携带 compressed-reduce capability；否则抛出明确的 `UnsupportedCollective`。

### 5.2 Reduction Contract

新增单一 reduction contract，负责：

- 校验 `sum`/`mean`；
- 保存 world size；
- 声明 transport 返回的是 sum、mean 还是未归一化 shard；
- 保证 mean 只执行一次除法；
- 为测试提供独立的 reference reduction。

DDP Hook、collective API、topology executor 和 reduce-scatter executor 共享该契约。

### 5.3 CapabilityReport

CapabilityReport 从 backend/registry/extension ABI 的真实状态生成，至少区分：

- `implemented`：存在实现；
- `available`：当前运行环境可加载；
- `verified`：命中经过测试的硬件和 shape 策略；
- `async_supported`：实现具备完整异步完成语义。

不再保留与实现状态矛盾的“not implemented yet”固定值。

## 6. 异步完成与资源生命周期

### 6.1 完成条件

CCDL Work/Future 只有在以下步骤全部完成后才允许 ready：

1. transport 完成；
2. dequant/reduce 完成；
3. error-feedback update 完成；
4. 输出 workspace 写入完成；
5. 完成 event 已记录，并与消费 stream 建立顺序关系。

`CompletionWork.get_future()` 不得直接暴露底层 transport Future。它必须返回一个代表完整 CCDL pipeline 的外层 Future；无法可靠构造时返回 `None`，由调用者使用 `wait()`。

### 6.2 同步与 fallback

- 真异步路径不得执行默认 CPU `cudaEventSynchronize`；
- 消费 stream 通过 `wait_event` 建立依赖；
- 只支持同步的 hierarchical prototype 必须明确报告 `async_supported=False`；
- runtime fallback 必须写入 ExecutionInfo 和计数器；
- 只捕获预期的 capability/extension 异常，未知异常必须传播。

### 6.3 计算与通信重叠

端到端 DDP 示例必须通过反向传播中的 bucket-ready 顺序启动通信，而不是在完整 backward 结束后批量通信。压缩、transport、反量化和反馈更新运行在具有明确 ownership 的 CUDA stream 上；后续计算或 optimizer 只能通过 event dependency 等待对应结果，不执行全设备同步。

重叠能力必须用时间线和派生指标验证：

```text
overlap_efficiency = (communication_ms + compute_ms - overlapped_ms)
                     / min(communication_ms, compute_ms)
```

报告同时记录 exposed communication time，避免只根据 API 返回 Future 就宣称存在重叠。负值按零报告，超过一的结果作为计时或同步错误处理。

### 6.4 Workspace ownership

send、recv、reduced、restored 和 residual workspace 在 completion event 前不能复用。Full-output restore 必须支持 caller-owned 或 pool-owned 连续输出，避免每步构造 tensor list 和 `torch.cat` 中间分配。

## 7. CUDA 安全性

所有公开 CUDA/C++ 入口必须：

- 校验 CUDA、dtype、shape、contiguous、device 和 workspace 容量；
- 使用输入或输出 tensor 的 `c10::cuda::CUDAGuard`；
- 从 guard 后的设备取得当前 stream；
- kernel launch 后执行 `C10_CUDA_KERNEL_LAUNCH_CHECK()`；
- 在错误信息中包含操作名和关键配置；
- 支持单进程多 GPU 下 tensor device 不等于进入函数前 current device 的情况。

自动生成的 quant/dequant API 必须从模板层统一满足这些要求，不能只修补单个生成文件。

## 8. Kernel 优化

正确性任务完成后依次进行：

1. 融合本地 reconstruction 和 residual update，避免为了 EF 额外生成完整临时张量；
2. dequant-reduce 输出直接写入复用 workspace；
3. full-output restore 使用连续 gather workspace，消除逐 rank 分配和 `torch.cat`；
4. 将支持超过 8 rank 的 fused reduction 设计为分段输入或分层归约，不在 kernel 参数中无限展开指针；
5. 根据 profiling 决定是否融合 mean，确保除法只发生一次。

每项优化必须同时满足：

- 与分步 FP32 reference 的误差门槛一致；
- sum/mean、奇数 numel、padding、不同 rank scale 均通过；
- workspace 生命周期测试通过；
- A6000 2/4 卡对应微基准不回退；
- 端到端训练吞吐和显存单独报告，不以通信微基准替代。

## 9. 测试策略

### 9.1 单元与协议测试

- 默认 DDP 路径选择 native NCCL；
- 未声明 capability 的 compressed all-reduce 被拒绝；
- 双 rank 不同 scale 的 bytewise all-reduce 回归测试；
- sum/mean 与 reference reduction 一致；
- local EF residual 与 `prepared - D(Q(prepared))` 一致；
- Future ready 时 completion callback 已执行；
- fallback reason 和计数器可观察；
- CapabilityReport 与 Registry 状态一致。

### 9.2 CUDA 测试

- 非 current-device tensor；
- kernel launch failure 在当前调用点报告；
- fused 与 unfused 结果对照；
- 2/4/8 rank、非整除 shape、FP16/BF16/FP32；
- stream/event ordering 和 workspace 延迟复用。

### 9.3 动态门禁

A6000 至少执行：

- 单机 2 卡、4 卡通信微基准；
- native DDP 与 CCDL full-output 同口径训练；
- ReducedShard 独立基准；
- 多随机种子短周期收敛验证；
- 一个完整训练周期的 loss、吞吐、显存与数值稳定性报告。

### 9.4 `examples/` 端到端训练示例

仓库新增独立 `examples/` 目录，至少包含：

- `examples/ddp_training.py`：可直接使用 `torchrun` 启动的端到端 DDP 示例；
- `examples/training/`：模型、数据、指标和启动配置等可复用组件；
- `examples/README.md`：native DDP、CCDL 同步压缩和 CCDL 异步重叠三种模式的命令；
- `examples/configs/`：A6000 2 卡和 4 卡可复现实验配置。

示例不得硬编码服务器、模型或 21 GB 数据集路径。命令行支持真实数据目录，并提供无需下载数据即可运行的 deterministic synthetic 模式。真实数据适配层只负责读取公开约定，不把业务模型代码复制进通信库。

每次运行输出机器可读 JSON，至少包含：

- world size、模型参数量、global/per-rank batch size；
- strategy、bit、group size、bucket size 和 error-feedback policy；
- warmup 后的 samples/s、step time P50/P95、峰值显存；
- communication、compute、overlapped 和 exposed communication 时间；
- overlap efficiency；
- 训练/验证 loss、梯度相对 L2、NaN/Inf 和 rank 参数一致性；
- capability、实际执行策略和 fallback reason。

端到端测试必须使用同模型、同初始权重、同数据顺序、同全局 batch 和同优化器超参数对比 native DDP 与 CCDL。异步模式只有在时间线证明通信与后续 backward compute 相交，并且端到端吞吐优于同语义同步压缩路径时，才标记为 overlap 有效。

示例代码纳入轻量 CPU/单进程测试；真实 2/4 卡训练由 A6000 动态门禁执行。

## 10. 实施与提交边界

按以下独立功能提交：

1. 默认策略与 compressed transport capability；
2. reduction contract 和 DDP mean 修复；
3. local error-feedback 语义；
4. 完整 Work/Future completion；
5. 统一异步 DDP pipeline 与可观察 fallback；
6. CapabilityReport 修复；
7. CUDA guard 与 launch check；
8. EF/reconstruction Kernel 融合；
9. output restore workspace 优化；
10. `examples/` 端到端训练与可复现配置；
11. A6000 计算通信重叠、性能与训练验证报告。

每个提交遵循 `<type>(<scope>): <subject>`，并在提交前运行对应测试。全量测试、Ruff 和 wheel 构建作为阶段性门禁。
