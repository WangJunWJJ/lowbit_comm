# CCDL Independent Communication Library Refactor Plan

## Architecture boundary

CCDL should become a communication execution layer:

```text
ParaScale
  -> communication plugin registry
  -> CCDL plugin adapter
  -> CCDL DDP hook
  -> compressed collective ops
  -> quantization CUDA kernels
```

CCDL should not own:

- training loops;
- model wrapping strategy;
- data loading;
- backend selection;
- workload benchmark decisions;
- checkpoint policy beyond CCDL-owned residual state.

## Phase 1: installable control-plane package

Goal:

```text
import ccdl_comm
```

Deliverables:

- stable `CompressionConfig`;
- stable `CapabilityReport`;
- stable ParaScale-facing plugin adapter;
- no-Torch tests;
- clean package metadata.

Status: started in this folder.

## Phase 1.5: safe CUDA import and generated-source guard

Goal:

```text
ParaScale can ask whether CCDL is usable without crashing on machines where
Torch, CUDA, or ccdl_cuda_ops are missing.
```

Deliverables:

- `ccdl_comm.cuda.load_cuda_extension()`;
- `ccdl_comm.build.ensure_generated_sources()`;
- no-Torch tests for missing extension and codegen behavior.

Status: implemented as control-plane scaffolding. CUDA compilation is not yet
migrated into the package build.

## Phase 1.6: quantization facade with extension fallback

Goal:

```text
ParaScale can call CCDL quantization through a stable Python facade and receive
a clear fallback error when the CUDA extension is unavailable.
```

Deliverables:

- `ccdl_comm.quantization.quantize_tensor()`;
- `ccdl_comm.quantization.dequantize_tensor()`;
- `CCDLUnavailableError` for planner/runtime fallback handling.

Status: implemented for extension-backed execution. CPU fallback quantization is
not implemented.

## Phase 1.7: quantized buffer sizing estimates

Goal:

```text
ParaScale can estimate compressed buffer size and padding without importing
Torch or loading the CUDA extension.
```

Deliverables:

- `ccdl_comm.quantization.estimate_quantized_size()`;
- group padding accounting for arbitrary bucket sizes;
- per-group metadata accounting for dtype and top-k mode.

Status: implemented for planning and benchmark metadata.

## Phase 2: quantization kernel migration

Goal:

```text
ccdl_comm.quantization.quantize(tensor, config)
ccdl_comm.quantization.dequantize(buffer, shape, config)
```

Required fixes from the original CCDL prototype:

- make code generation deterministic in the build;
- avoid undefined symbols when generated `.cu` files are missing;
- unify `pyproject.toml` and build metadata;
- expose CPU-safe import paths when CUDA extension is unavailable;
- preserve `linear 8-bit group_size=64` as the first supported production path.

## Phase 3: compressed collective ops

Goal:

```python
work = compressed_all_reduce(tensor, op="mean", config=config, async_op=True)
work.wait()
```

Required semantics:

- PyTorch-like blocking and async API;
- safe tensor and quantized-buffer lifetime;
- clear fallback to `torch.distributed` when compression is disabled;
- rank-consistency debug checks.

## Phase 4: DDP communication hook

Goal:

```python
hook, state = build_ddp_comm_hook(config)
model.register_comm_hook(state, hook)
```

Hard requirements:

- hook signature compatible with PyTorch DDP;
- return `torch.futures.Future[Tensor]`;
- bucket padding/unpadding for arbitrary bucket sizes;
- optional error-feedback residual;
- residual state checkpoint helpers.

## Phase 5: ParaScale integration

ParaScale should consume CCDL through a plugin contract:

```python
plugin = CCDLCommunicationPlugin()
decision = plugin.plan(context, config)
```

ParaScale remains responsible for:

- selecting `native_ddp`;
- rejecting unsupported CPU/NPU/FSDP contexts in early versions;
- warmup and fallback;
- benchmark evidence and tuner reports;
- workload-level correctness checks.

## Phase 6: benchmark gate

Required evidence before production enablement:

- quant/dequant latency;
- compression ratio;
- all-reduce latency;
- relative L2 error;
- rank weight consistency;
- loss curve comparison;
- real workload benchmark under the same data/model/batch/precision contract.

First production candidate:

```text
CUDA + native DDP + linear 8-bit + group_size 64 + topk 0 + error feedback
```

Experimental only:

- 4-bit;
- topk;
- stochastic rounding;
- compact layout;
- FSDP reduce-scatter hooks;
- non-CUDA backends.
