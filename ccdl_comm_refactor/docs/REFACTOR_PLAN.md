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
