# CCDL Correctness, Async Semantics, and Kernel Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复重构版 CCDL 的 P0/P1 正确性和完成语义问题，在严格数值门禁后优化 CUDA Kernel，并提供能够验证真实计算—通信重叠的端到端训练示例。

**Architecture:** 默认 DDP 使用 native NCCL；压缩 transport 通过不可变 capability 显式证明其理解 CCDL payload。统一 reduction contract、本地 error-feedback 和完整 pipeline Future 后，再将本地重建与 residual 更新融合进 CUDA Kernel，并用连续 workspace 优化 full-output restore。

**Tech Stack:** Python 3.10+、PyTorch Distributed、NCCL、CUDA C++、pybind11、pytest、Ruff、torchrun。

## Global Constraints

- 默认 `create_ddp_comm_hook()` 必须使用 `strategy="native_nccl"`、`reduce="mean"`。
- 普通 `torch.distributed.all_reduce(uint8_payload)` 永远不是合法 compressed-reduce transport。
- `mean` 只能归一化一次；所有 full-output rank 结果一致。
- Error Feedback 必须使用 `prepared_i - D(Q(prepared_i))` 本地残差。
- Work/Future ready 必须覆盖 transport、dequant/reduce、EF、workspace 写入和 CUDA event ordering。
- 未验证配置默认 fallback 到 native NCCL，不得静默改变数学语义。
- 每项生产代码修改必须先运行对应失败测试，再进行最小实现。
- 每个独立任务按 `<type>(<scope>): <subject>` 提交。
- CUDA 优化必须在 A6000 2/4 卡上不低于修改前同口径通信性能。

---

## File Structure

- `ccdl_comm/reduction.py`：唯一的 sum/mean 数学契约与归一化规则。
- `ccdl_comm/communication/transport_capability.py`：压缩 transport capability 及校验。
- `ccdl_comm/communication/ddp_hook.py`：DDP 策略绑定，不再拥有隐式归约语义。
- `ccdl_comm/quantization/error_feedback.py`：本地 residual 状态。
- `ccdl_comm/work.py`：完整 pipeline Future。
- `ccdl_comm/communication/async_pipeline.py`：event-driven DDP completion。
- `ccdl_comm/capability.py`：从 backend 状态构建真实能力报告。
- `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`：融合全局归约、本地重建和 EF 更新。
- `ccdl_comm/communication/reduce_scatter_transport.py`：连续 full-output restore workspace。
- `examples/ddp_training.py`：可直接运行的端到端入口。
- `examples/training/`：示例模型、数据、计时与结果模式。

### Task 1: 禁用不安全 compressed bytewise all-reduce 并切换安全默认值

**Files:**
- Create: `ccdl_comm/communication/transport_capability.py`
- Modify: `ccdl_comm/communication/ddp_hook.py:48-355`
- Modify: `ccdl_comm/communication/collectives.py`
- Modify: `ccdl_comm/communication/__init__.py`
- Test: `tests/test_transport_capability.py`
- Test: `tests/test_ddp_comm_hook.py`

**Interfaces:**
- Produces: `CompressedTransportCapability`、`capability_for(transport: object) -> CompressedTransportCapability | None`、`require_compressed_transport(transport, *, collective, config, dtype, output_layout)`。
- Produces: `create_ddp_comm_hook(config: CompressionConfig, *, strategy: str = "native_nccl", reduce: str = "mean", native_all_reduce: Callable[[Any, str], Any] | None = None)`。
- Consumes: `CompressionConfig`、现有 `native_all_reduce` callable。

- [ ] **Step 1: 写 capability 和默认策略失败测试**

```python
def test_ddp_hook_defaults_to_native_nccl():
    calls = []
    hook = create_ddp_comm_hook(
        CompressionConfig(error_feedback=False),
        native_all_reduce=lambda tensor, op: calls.append(op) or tensor,
    )
    result = hook(None, FakeBucket(FakeTensor([1.0]))).wait()
    assert result.values == [1.0]
    assert calls == ["mean"]


def test_compressed_all_reduce_rejects_transport_without_payload_capability():
    with pytest.raises(UnsupportedCollective, match="compressed payload"):
        create_ddp_comm_hook(
            CompressionConfig(),
            strategy="all_reduce",
            all_reduce=lambda payload, op: payload,
        )
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_transport_capability.py tests/test_ddp_comm_hook.py -q`

Expected: 默认策略仍为 `all_reduce`，且无 capability 的 callable 未被拒绝。

- [ ] **Step 3: 实现不可变 transport capability**

```python
@dataclass(frozen=True, slots=True)
class CompressedTransportCapability:
    codec: str
    collectives: frozenset[str]
    bits: frozenset[int]
    group_sizes: frozenset[int]
    dtypes: frozenset[str]
    output_layouts: frozenset[str]
    supports_async: bool = False


def require_compressed_transport(
    transport: object,
    *,
    collective: str,
    config: CompressionConfig,
    dtype: str,
    output_layout: str,
) -> CompressedTransportCapability:
    capability = getattr(transport, "ccdl_compressed_capability", None)
    if not isinstance(capability, CompressedTransportCapability):
        raise UnsupportedCollective(
            collective,
            reason="transport does not declare CCDL compressed payload capability",
        )
    capability.require_support(
        collective=collective,
        bit=config.bit,
        group_size=config.group_size,
        dtype=dtype,
        output_layout=output_layout,
    )
    return capability
```

将 DDP 默认策略改为 `native_nccl`；`all_reduce` 分支不再调用 `make_torch_all_reduce()`，且只接受 capability 校验通过的显式 transport。

- [ ] **Step 4: 验证 GREEN 与相关回归**

Run: `python -m pytest tests/test_transport_capability.py tests/test_ddp_comm_hook.py tests/test_torch_transport.py -q`

Expected: 全部通过；原生 tensor transport 测试保留，但不再被当作 compressed transport。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/communication tests/test_transport_capability.py tests/test_ddp_comm_hook.py
git commit -m "fix(ddp): gate compressed transports and use native default"
```

### Task 2: 统一 reduction contract 并修复 DDP mean

**Files:**
- Create: `ccdl_comm/reduction.py`
- Modify: `ccdl_comm/communication/ddp.py`
- Modify: `ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm/collectives/all_reduce.py`
- Modify: `ccdl_comm/collectives/reduce_scatter.py`
- Test: `tests/core/test_reduction.py`
- Test: `tests/test_ddp_comm_hook.py`

**Interfaces:**
- Produces: `ReductionContract(op: Literal["sum", "mean"], world_size: int, transport_output: Literal["sum", "mean"])`。
- Produces: `ReductionContract.normalize(tensor)` 和 `transport_op`。
- Consumes: transport 返回结果和显式 world size。

- [ ] **Step 1: 写 sum/mean 和单次归一化失败测试**

```python
def test_mean_contract_normalizes_transport_sum_once():
    contract = ReductionContract(op="mean", world_size=4, transport_output="sum")
    assert contract.normalize(FakeTensor([8.0])).values == [2.0]


def test_mean_contract_does_not_normalize_transport_mean_again():
    contract = ReductionContract(op="mean", world_size=4, transport_output="mean")
    assert contract.normalize(FakeTensor([2.0])).values == [2.0]
```

增加 DDP 测试，证明 `reduce="mean"` 不再固定传 `sum` 后直接返回。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/core/test_reduction.py tests/test_ddp_comm_hook.py -q`

Expected: `ReductionContract` 不存在，原 DDP processor mean 对照失败。

- [ ] **Step 3: 最小实现 reduction contract**

```python
@dataclass(frozen=True, slots=True)
class ReductionContract:
    op: Literal["sum", "mean"]
    world_size: int
    transport_output: Literal["sum", "mean"] = "sum"

    @property
    def transport_op(self) -> str:
        return "sum" if self.op == "mean" else self.op

    def normalize(self, tensor: Any) -> Any:
        if self.op == "mean" and self.transport_output == "sum":
            return tensor / self.world_size
        return tensor
```

`DDPBucketProcessor.process()` 接收 `reduction: ReductionContract`，不得再固定 `op="sum"`。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/core/test_reduction.py tests/test_ddp_comm_hook.py tests/test_compressed_collective_async.py tests/test_reduce_scatter_api.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/reduction.py ccdl_comm/communication/ddp.py ccdl_comm/communication/ddp_hook.py ccdl_comm/collectives tests
git commit -m "fix(collective): enforce explicit reduction semantics"
```

### Task 3: 将 Error Feedback 改为本地量化误差

**Files:**
- Modify: `ccdl_comm/quantization/error_feedback.py`
- Modify: `ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm/communication/ddp.py`
- Modify: `ccdl_comm/quantization/codec.py`
- Test: `tests/test_error_feedback.py`
- Test: `tests/test_ddp_comm_hook.py`
- Create: `tests/distributed/ddp_local_error_feedback_oracle.py`

**Interfaces:**
- Produces: `ErrorFeedbackState.update_local(key, *, prepared, local_restored)`。
- DDP compressed path consumes `local_payload` and reconstructs only that payload for residual update。
- 暂停使用现有“global restored 更新 residual”的融合入口，直到 Task 8 提供正确 Kernel。

- [ ] **Step 1: 写不同 rank 梯度的本地 residual 失败测试**

```python
def test_ddp_feedback_uses_local_reconstruction_not_global_mean():
    local_prepared = FakeTensor([4.0])
    local_restored = FakeTensor([3.5])
    global_mean = FakeTensor([2.0])
    state = ErrorFeedbackState()
    state.update_local("bucket", prepared=local_prepared, local_restored=local_restored)
    assert state.get("bucket").values == [0.5]
    assert state.get("bucket").values != [2.0]
```

Hook 测试注入两个不同 payload，断言反馈 update 接收本地解码值而不是全局输出。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_error_feedback.py tests/test_ddp_comm_hook.py -q`

Expected: `update_local` 不存在或 Hook residual 仍等于 prepared-global。

- [ ] **Step 3: 实现本地 residual 路径**

```python
def update_local(self, key: Hashable, *, prepared: Any, local_restored: Any) -> None:
    self._residuals[key] = _safe_detached_clone(prepared - local_restored)
```

在 quantize 后保留 `local_payload`；policy 要求更新时，通过 `active_dequantize(local_payload, tuple(prepared.shape), config, active_dtype)` 计算本地 reconstruction，再更新 residual。删除 DDP 对旧 global-restored EF 融合入口的调用。

- [ ] **Step 4: 验证 GREEN 和双 rank oracle**

Run: `python -m pytest tests/test_error_feedback.py tests/test_ddp_comm_hook.py -q`

Remote run: `torchrun --standalone --nproc-per-node=2 tests/distributed/ddp_local_error_feedback_oracle.py`

Expected: 两个 rank 输出相同 global gradient；各 rank residual 分别等于自己的本地量化误差。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/quantization ccdl_comm/communication tests
git commit -m "fix(quantization): use local reconstruction for error feedback"
```

### Task 4: 修复完整 pipeline Future 语义

**Files:**
- Modify: `ccdl_comm/work.py`
- Modify: `ccdl_comm/communication/cuda_completion.py`
- Modify: `ccdl_comm/communication/async_pipeline.py`
- Modify: `ccdl_comm/communication/async_shard_pipeline.py`
- Test: `tests/core/test_work_protocol.py`
- Test: `tests/test_async_bucket_pipeline.py`
- Test: `tests/test_async_shard_pipeline.py`

**Interfaces:**
- Produces: `CompletionWork(result, *, handle=None, complete=None, completion=None, resources=(), future_factory: Callable[[], Any] | None = None)`。
- `get_future()` 返回 CCDL 外层 Future，或在无法可靠完成时返回 `None`。
- `wait()`、callback 和 Future 共享一次性 terminal state。

- [ ] **Step 1: 写 Future 不得提前 ready 的失败测试**

```python
def test_completion_work_future_waits_for_deferred_callback():
    handle = ControlledHandle()
    work = CompletionWork(None, handle=handle, complete=lambda: "decoded")
    future = work.get_future()
    handle.finish_transport()
    assert not future.done()
    assert work.wait() == "decoded"
    assert future.wait() == "decoded"
```

同时覆盖 callback 异常传播和 callback 只执行一次。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/core/test_work_protocol.py tests/test_async_bucket_pipeline.py tests/test_async_shard_pipeline.py -q`

Expected: 当前 `get_future()` 在 transport 完成后提前 ready。

- [ ] **Step 3: 实现外层 Future 和 terminal state**

CompletionWork 在构造时创建外层 Future；底层 Future callback 只触发 `_finish_pipeline()`，该方法顺序执行 handle completion、`complete`、completion event，并设置结果或异常。无 Future 工厂时 `get_future()` 返回 `None`。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/core/test_work_protocol.py tests/core/test_work_execution_info.py tests/test_async_bucket_pipeline.py tests/test_async_shard_pipeline.py -q`

Expected: 全部通过且不存在重复 callback。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/work.py ccdl_comm/communication/cuda_completion.py ccdl_comm/communication/async_pipeline.py ccdl_comm/communication/async_shard_pipeline.py tests
git commit -m "fix(async): complete futures after the full CCDL pipeline"
```

### Task 5: 统一 DDP 异步 event 链和可观察 fallback

**Files:**
- Modify: `ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm/communication/async_pipeline.py`
- Modify: `ccdl_comm/execution_info.py`
- Modify: `ccdl_comm/cuda/executors.py`
- Test: `tests/test_ddp_comm_hook.py`
- Test: `tests/test_async_bucket_pipeline.py`
- Test: `tests/core/test_work_execution_info.py`
- Create: `tests/distributed/ddp_async_event_oracle.py`

**Interfaces:**
- Produces: `FallbackRecord(reason: str, from_path: str, to_path: str)` in execution metadata/counters。
- Produces: `AsyncBucketPipeline(*, gather_work: Any, future: Any, dequantize_reduce: Callable[[Any], Any], update_feedback: Callable[[Any], None], advance_policy: Callable[[], None], completion_manager: CudaCompletionManager | None = None, synchronize_completion: bool = False)` 默认真异步语义。
- Consumes: Task 4 完整 Future。

- [ ] **Step 1: 写无 CPU synchronize 与异常可见性失败测试**

```python
def test_async_pipeline_orders_stream_without_cpu_synchronize():
    completion = FakeCompletion()
    pipeline = make_pipeline(completion=completion)
    pipeline.run()
    assert completion.wait_stream_calls == 1
    assert completion.synchronize_calls == 0


def test_unexpected_fused_kernel_error_is_not_silently_swallowed():
    with pytest.raises(RuntimeError, match="kernel failed"):
        run_hook_with_fused_error(RuntimeError("kernel failed"))
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_async_bucket_pipeline.py tests/test_ddp_comm_hook.py tests/core/test_work_execution_info.py -q`

Expected: 默认同步计数大于零，或未知异常被 `except Exception: pass` 吞掉。

- [ ] **Step 3: 实现 event 链和窄异常 fallback**

只捕获 `CCDLUnavailableError`、`UnsupportedCollective` 和明确的 capability 异常；记录 fallback 后选择安全 executor。未知 `RuntimeError` 传播。将 consumer stream dependency 绑定到 completion event，不调用 CPU synchronize。

- [ ] **Step 4: 验证 GREEN 和 CUDA oracle**

Run: `python -m pytest tests/test_async_bucket_pipeline.py tests/test_ddp_comm_hook.py tests/core/test_work_execution_info.py -q`

Remote run: `torchrun --standalone --nproc-per-node=2 tests/distributed/ddp_async_event_oracle.py`

Expected: Future 结果可直接由 optimizer stream 消费，无全设备同步，fallback reason 可读取。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/communication ccdl_comm/execution_info.py ccdl_comm/cuda/executors.py tests
git commit -m "fix(async): unify DDP event ordering and fallback reporting"
```

### Task 6: 让 CapabilityReport 反映真实 backend 能力

**Files:**
- Modify: `ccdl_comm/capability.py`
- Modify: `ccdl_comm/backend.py`
- Modify: `ccdl_comm/cuda/backend.py`
- Modify: `ccdl_comm/plugin.py`
- Test: `tests/test_capability.py`
- Test: `tests/conformance/test_cuda_backend.py`

**Interfaces:**
- Extends: `BackendCapabilities` with `verified_strategies` and `async_strategies`。
- Produces: `CapabilityReport.from_backend_capabilities(capabilities: BackendCapabilities, *, cuda: bool, torch_version: str | None, cuda_arch: str | None)`。
- `detect()` 只负责环境检测，再委托 CUDA backend 构造真实能力。

- [ ] **Step 1: 写能力一致性失败测试**

```python
def test_detect_reports_implemented_compressed_collectives():
    report = detect(import_torch=fake_cuda_torch, import_extension=fake_extension)
    assert report.quantize
    assert report.compressed_collectives
    assert report.ddp_hook
    assert "not implemented yet" not in report.warnings
```

增加 verified 与 available 分离测试。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_capability.py tests/conformance/test_cuda_backend.py -q`

Expected: 固定的 `compressed_collectives=False` 导致失败。

- [ ] **Step 3: 实现能力归一化**

将 Registry/backend 声明转为 CapabilityReport；`verified` 只对策略表命中的组合为真，implemented 不等于 verified。同步 hierarchical 不得报告 async。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_capability.py tests/conformance/test_cuda_backend.py tests/cuda/test_strategy_table.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/backend.py ccdl_comm/capability.py ccdl_comm/cuda/backend.py ccdl_comm/plugin.py tests
git commit -m "fix(capability): report implemented and verified CUDA paths"
```

### Task 7: 加固 CUDA device、stream、launch 和 ABI 诊断

**Files:**
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm/csrc/quantization/quant_pack_kernel.cu`
- Modify: `ccdl_comm/csrc/quantization/gen_code_quant.py`
- Modify: `ccdl_comm/csrc/quantization/gen_code_dequant.py`
- Modify generated: `ccdl_comm/csrc/quantization/gen_quant_api.cu`
- Modify generated: `ccdl_comm/csrc/quantization/gen_dequant_api.cu`
- Modify: `ccdl_comm/cuda/loader.py`
- Test: `tests/test_quantization_codec.py`
- Test: `tests/test_codegen.py`
- Test: `tests/cuda/test_fused_dequant_reduce_mean_ef.py`
- Create: `tests/distributed/cuda_non_current_device_oracle.py`

**Interfaces:**
- Every native entry uses `c10::cuda::CUDAGuard guard(tensor.device())` before `get_current_cuda_stream()`。
- Every launch ends with `C10_CUDA_KERNEL_LAUNCH_CHECK()`。
- Loader diagnostics include extension ABI、Torch version 和 CUDA runtime version。

- [ ] **Step 1: 写源码契约和多设备失败测试**

```python
def test_generated_cuda_entry_uses_guard_and_launch_check():
    source = generate_quant_source()
    assert "c10::cuda::CUDAGuard" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
```

CUDA oracle 在 device 0 为 current device 时，对 device 1 tensor 调用 quant/dequant/EF，并与 PyTorch reference 比较。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_codegen.py tests/test_quantization_codec.py -q`

Expected: 生成源码缺少 guard 或 launch check。

- [ ] **Step 3: 修改模板和手写 Kernel**

在模板中引入 `<c10/cuda/CUDAGuard.h>` 与 `<c10/cuda/CUDAException.h>`，从 tensor device 建 guard 后获取 stream，并在所有 launch 后检查。重新生成源码，不手工维护生成差异。

- [ ] **Step 4: 本地与 A6000 验证**

Run: `python -m pytest tests/test_codegen.py tests/test_quantization_codec.py tests/test_pybind_exports.py -q`

Remote run: `python -m pytest tests/cuda/test_fused_dequant_reduce_mean_ef.py -q && torchrun --standalone --nproc-per-node=2 tests/distributed/cuda_non_current_device_oracle.py`

Expected: 全部通过；错误在当前 API 调用点报告。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/csrc ccdl_comm/cuda/loader.py tests
git commit -m "fix(cuda): guard devices and surface kernel launch failures"
```

### Task 8: 融合本地 reconstruction、global dequant-reduce 和 EF update

**Files:**
- Modify: `ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm/csrc/pybind.cpp`
- Modify: `ccdl_comm/quantization/codec.py`
- Modify: `ccdl_comm/communication/ddp_hook.py`
- Test: `tests/cuda/test_fused_dequant_reduce_mean_ef.py`
- Test: `tests/test_quantization_codec.py`
- Modify: `tests/benchmarks/fused_dequant_executor_gate.py`

**Interfaces:**
- Produces native symbol:

```cpp
bool inplace_dequantize_reduce_update_local_error_feedback(
    std::vector<torch::Tensor> inputs,
    int64_t local_input_index,
    torch::Tensor prepared,
    torch::Tensor restored,
    torch::Tensor residual,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    int64_t divisor);
```

- Produces Python wrapper `inplace_dequantize_reduce_update_local_feedback(buffers, local_input_index, prepared, restored, residual, config, *, extension_status, reduce) -> bool`。
- Consumes rank-ordered gathered payloads and local group rank。

- [ ] **Step 1: 写 fused/unfused 数值等价失败测试**

对每个 rank 使用不同 scale，计算：

```python
expected_global = sum(dequant(payload) for payload in payloads) / world_size
expected_residual = prepared - dequant(payloads[local_rank])
```

断言 fused output/residual 分别接近两个 reference，而不是 `prepared-expected_global`。覆盖 FP16/BF16/FP32、奇数 numel、sum/mean、local rank 0/末位。

- [ ] **Step 2: 验证 RED**

Remote run: `python -m pytest tests/cuda/test_fused_dequant_reduce_mean_ef.py -q`

Expected: 当前 fused residual 使用 global restored，测试失败。

- [ ] **Step 3: 实现单 launch Kernel**

每个输出 index 在同一循环中累加各 payload 解码值，同时保存 `inputs[local_input_index]` 的解码值；写入 `restored=global*inv_divisor` 和 `residual=prepared-local_restored`。禁止分配 local-restored 中间 tensor。

- [ ] **Step 4: 验证 GREEN 与性能门禁**

Remote run: `python -m pytest tests/cuda/test_fused_dequant_reduce_mean_ef.py tests/test_quantization_codec.py -q`

Remote run: `torchrun --standalone --nproc-per-node=2 tests/benchmarks/fused_dequant_executor_gate.py --dtype fp16 --numel 8388608`

Remote run: `torchrun --standalone --nproc-per-node=4 tests/benchmarks/fused_dequant_executor_gate.py --dtype fp16 --numel 8388608`

Expected: 数值测试通过；P50 不高于修改前同口径基准，额外 local reconstruction allocation 为零。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/csrc ccdl_comm/quantization/codec.py ccdl_comm/communication/ddp_hook.py tests
git commit -m "perf(cuda): fuse local error feedback into dequant reduce"
```

### Task 9: 复用 full-output restore 连续 workspace

**Files:**
- Modify: `ccdl_comm/communication/reduce_scatter_transport.py`
- Modify: `ccdl_comm/cuda/workspace.py`
- Modify: `ccdl_comm/cuda/compiler.py`
- Test: `tests/test_reduce_scatter_transport.py`
- Test: `tests/cuda/test_cuda_workspace_pool.py`
- Test: `tests/cuda/test_cuda_backend_compile.py`

**Interfaces:**
- Adds: `allocate_full_output_workspace(tensor, world_size) -> Tensor` callback。
- Full restore prefers `dist.all_gather_into_tensor(contiguous_output, shard)`。
- Output lease remains owned until completion event and consumer release。

- [ ] **Step 1: 写无 list/cat 和延迟复用失败测试**

```python
def test_full_restore_uses_contiguous_caller_workspace():
    output = FakeTensor.empty(16)
    transport = make_torch_compressed_reduce_scatter_all_gather(
        shard_transport=fake_shard_transport,
        allocate_full_output_workspace=lambda *_: output,
        import_module=fake_torch_modules,
    )
    result = transport(
        FakeTensor.range(16),
        config=CompressionConfig(),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=FakeExtensionStatus(),
    )
    assert result is output
    assert fake_dist.all_gather_into_tensor_calls == 1
    assert fake_torch.cat_calls == 0
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_reduce_scatter_transport.py tests/cuda/test_cuda_workspace_pool.py -q`

Expected: 当前实现分配 list 并调用 `torch.cat`。

- [ ] **Step 3: 实现连续 restore workspace**

按 bucket shape/dtype/world size 缓存 padded contiguous output；使用 `all_gather_into_tensor`，最后返回 logical view。异步时 workspace lease 绑定到 CompletionWork resources。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_reduce_scatter_transport.py tests/cuda/test_cuda_workspace_pool.py tests/cuda/test_cuda_backend_compile.py -q`

Expected: 全部通过且 in-flight workspace 不会提前复用。

- [ ] **Step 5: 提交**

```bash
git add ccdl_comm/communication/reduce_scatter_transport.py ccdl_comm/cuda/workspace.py ccdl_comm/cuda/compiler.py tests
git commit -m "perf(workspace): reuse contiguous full-output restore buffers"
```

### Task 10: 建立可运行的端到端训练 examples

**Files:**
- Create: `examples/ddp_training.py`
- Create: `examples/training/__init__.py`
- Create: `examples/training/config.py`
- Create: `examples/training/data.py`
- Create: `examples/training/model.py`
- Create: `examples/training/metrics.py`
- Create: `examples/configs/a6000_2gpu.json`
- Create: `examples/configs/a6000_4gpu.json`
- Create: `examples/README.md`
- Create: `tests/examples/test_training_config.py`
- Create: `tests/examples/test_training_metrics.py`
- Create: `tests/examples/test_ddp_training_cli.py`

**Interfaces:**
- Produces CLI modes: `native_ddp`、`ccdl_sync`、`ccdl_async`。
- Produces `TrainingResult.to_dict()` with throughput、latency、memory、loss、overlap、capability and fallback fields。
- Consumes `--data-root` or `--synthetic`，不硬编码路径。

- [ ] **Step 1: 写 CLI、配置和 JSON schema 失败测试**

```python
def test_parser_supports_three_comparable_modes():
    parser = build_parser()
    for mode in ("native_ddp", "ccdl_sync", "ccdl_async"):
        args = parser.parse_args(["--mode", mode, "--synthetic"])
        assert args.mode == mode


def test_result_contains_overlap_and_correctness_fields():
    result = minimal_training_result().to_dict()
    assert result["timing"]["overlap_efficiency"] == 0.5
    assert "rank_parameters_consistent" in result["correctness"]
    assert "fallback_reason" in result["execution"]
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/examples -q`

Expected: `examples.training` 不存在。

- [ ] **Step 3: 实现 deterministic 示例框架**

模型使用可配置深度/宽度的 MLP，默认约 4496 万参数；synthetic dataset 使用 index 派生固定输入和标签，保证各模式同数据顺序。CLI 初始化 process group、DDP、Hook、optimizer，warmup 后采样并由 rank 0 写 JSON。

- [ ] **Step 4: 验证 GREEN 和单机 smoke**

Run: `python -m pytest tests/examples -q`

Run: `python examples/ddp_training.py --mode native_ddp --synthetic --steps 2 --device cpu --output dist/example-smoke.json`

Expected: 测试通过，JSON schema 完整。

- [ ] **Step 5: 提交**

```bash
git add examples tests/examples
git commit -m "feat(examples): add reproducible end-to-end DDP training"
```

### Task 11: 在 example 中证明真实计算—通信重叠

**Files:**
- Create: `examples/training/overlap.py`
- Modify: `examples/ddp_training.py`
- Modify: `examples/training/metrics.py`
- Modify: `examples/README.md`
- Test: `tests/examples/test_overlap_metrics.py`
- Test: `tests/examples/test_ddp_training_cli.py`
- Create: `tests/distributed/ddp_overlap_timeline.py`

**Interfaces:**
- Produces: `OverlapMeasurement(communication_ms, compute_ms, overlapped_ms, exposed_communication_ms)`。
- Produces: `overlap_efficiency()`，范围 `[0, 1]`，超界测量抛出 `InvalidOverlapMeasurement`。
- Consumes CUDA events/NVTX ranges and Task 5 full pipeline Future。

- [ ] **Step 1: 写 overlap 计算和无伪异步失败测试**

```python
def test_overlap_efficiency_uses_executed_timeline():
    measurement = OverlapMeasurement(
        communication_ms=4.0,
        compute_ms=6.0,
        overlapped_ms=8.0,
        exposed_communication_ms=2.0,
    )
    assert measurement.overlap_efficiency() == 0.5


def test_async_label_requires_timeline_intersection():
    result = classify_overlap(future_returned=True, timeline_intersection_ms=0.0)
    assert result == "not_overlapped"
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/examples/test_overlap_metrics.py tests/examples/test_ddp_training_cli.py -q`

Expected: overlap 模块不存在。

- [ ] **Step 3: 实现时间线与 exposed communication 测量**

为 backward compute、bucket communication 和 optimizer dependency 建立 CUDA event/NVTX 区间；只在 warmup 后统计。CCDL async 使用独立 comm stream，主 stream 只等待对应 bucket completion event，不调用 `torch.cuda.synchronize()` 作为执行语义；同步只允许出现在测量边界。

- [ ] **Step 4: 验证 GREEN 与 A6000 timeline oracle**

Run: `python -m pytest tests/examples -q`

Remote run: `torchrun --standalone --nproc-per-node=2 tests/distributed/ddp_overlap_timeline.py`

Expected: timeline 存在正交叠区间，效率在 `[0, 1]`，输出可被 optimizer 安全消费。

- [ ] **Step 5: 提交**

```bash
git add examples tests/examples tests/distributed/ddp_overlap_timeline.py
git commit -m "feat(examples): measure real DDP compute communication overlap"
```

### Task 12: A6000 2/4 卡端到端性能和收敛闭环

**Files:**
- Create: `tests/benchmarks/run_e2e_overlap_gate.py`
- Create: `tests/test_e2e_overlap_gate.py`
- Create: `tests/benchmarks/reports/correctness_async_kernel_20260805/README.md`
- Create raw JSON under: `tests/benchmarks/reports/correctness_async_kernel_20260805/raw/`
- Modify: `examples/README.md`

**Interfaces:**
- Produces gate comparing `native_ddp`、`ccdl_sync`、`ccdl_async` under identical workload。
- Gate requires correctness before performance and records non-passing results without hiding them。

- [ ] **Step 1: 写 benchmark gate 失败测试**

```python
def test_gate_rejects_speedup_with_rank_mismatch():
    with pytest.raises(GateFailure, match="rank parameters"):
        evaluate_run(candidate(speedup=1.2, rank_parameters_consistent=False))


def test_gate_requires_async_to_beat_sync_compression():
    with pytest.raises(GateFailure, match="overlap benefit"):
        evaluate_run(candidate(async_throughput=100.0, sync_throughput=101.0))
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_e2e_overlap_gate.py -q`

Expected: gate 模块不存在。

- [ ] **Step 3: 实现 gate 和报告聚合**

Gate 顺序：NaN/Inf、rank 一致性、梯度误差、loss 阈值、Future/event 语义、异步相对同步收益、最后才比较 native DDP。单次运行不宣称最终精度等价。

- [ ] **Step 4: 运行本地全量门禁**

Run: `python -m pytest -q`

Run: `python -m ruff check ccdl_comm tests examples`

Run: `python -m build --wheel --no-isolation packages/ccdl-core`

Expected: 全部通过。

- [ ] **Step 5: 运行 A6000 2/4 卡同口径测试**

```bash
torchrun --standalone --nproc-per-node=2 examples/ddp_training.py --config examples/configs/a6000_2gpu.json --mode native_ddp
torchrun --standalone --nproc-per-node=2 examples/ddp_training.py --config examples/configs/a6000_2gpu.json --mode ccdl_sync
torchrun --standalone --nproc-per-node=2 examples/ddp_training.py --config examples/configs/a6000_2gpu.json --mode ccdl_async
torchrun --standalone --nproc-per-node=4 examples/ddp_training.py --config examples/configs/a6000_4gpu.json --mode native_ddp
torchrun --standalone --nproc-per-node=4 examples/ddp_training.py --config examples/configs/a6000_4gpu.json --mode ccdl_sync
torchrun --standalone --nproc-per-node=4 examples/ddp_training.py --config examples/configs/a6000_4gpu.json --mode ccdl_async
```

随后使用 21 GB 真实数据集执行相同三模式完整周期，报告吞吐、P50/P95、显存、overlap efficiency、loss、收敛步数和 fallback。

- [ ] **Step 6: 提交动态证据**

```bash
git add tests/benchmarks examples/README.md tests/test_e2e_overlap_gate.py
git commit -m "test(benchmark): validate A6000 overlap and training convergence"
```

### Task 13: 最终回归、文档和兼容矩阵

**Files:**
- Modify: `README.md`
- Modify: `docs/INDEPENDENT_DDP_USAGE.md`
- Modify: `docs/SOFTWARE_DESIGN_ZH.md`
- Modify: `docs/SOFTWARE_REQUIREMENTS_ZH.md`
- Modify: `packages/ccdl-cuda/pyproject.toml`
- Modify: `packages/ccdl-ascend/pyproject.toml`
- Test: `tests/test_package_build.py`
- Test: `tests/test_package_ownership.py`

**Interfaces:**
- Documents safe defaults、explicit compressed opt-in、EF equation、Future semantics、verified hardware matrix。
- Native package runtime checks report Torch/CUDA/CANN ABI mismatch before symbol use。

- [ ] **Step 1: 写文档/API/ABI 契约失败测试**

增加测试断言 README 示例使用 `strategy="native_nccl"`；wheel metadata 包含受支持 Torch minor-version marker 或运行时 compatibility metadata；旧的“not implemented yet”文案不存在。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_package_build.py tests/test_package_ownership.py tests/test_capability.py -q`

Expected: 旧 metadata 或文档断言失败。

- [ ] **Step 3: 更新文档和兼容性声明**

记录已验证 Torch/CUDA/A6000 组合；wheel 名称和版本保持现有拆包结构。若 Python metadata 无法表达 CUDA ABI，则依赖保持合理范围，并由 loader 对 Torch/CUDA/extension ABI 进行硬校验和明确报错。

- [ ] **Step 4: 最终验证**

Run: `python -m pytest -q`

Run: `python -m ruff check ccdl_comm tests examples`

Run: `python -m build --wheel --no-isolation packages/ccdl-core`

Run on A6000 image: build `ccdl-cuda` wheel, install into clean container, execute CUDA smoke and 2-rank DDP oracle。

Expected: 所有门禁通过；报告明确列出任何未达到的性能目标，不以 fallback 结果冒充 compressed 结果。

- [ ] **Step 5: 提交**

```bash
git add README.md docs packages tests
git commit -m "docs(release): publish safe usage and compatibility matrix"
```
