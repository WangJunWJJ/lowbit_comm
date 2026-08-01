# Fused Dequant Reduce Mean Error Feedback Design

## Goal

Move the hot `dequant -> reduce -> mean -> error-feedback update` path from C++ orchestration into a CUDA kernel for the common CCDL communication case, while preserving safe fallback for unsupported shapes and quantization modes.

## Scope

This phase targets the current validated CUDA fast path:

- `group_size == 64`
- `bit == 8`
- `topk == 0`
- `quant_type == Linear`
- input count `1..8`
- output dtype `fp16`, `bf16`, or `fp32`
- compact and non-compact payload layouts

Unsupported configurations keep using the existing native/Python fallback path. This avoids changing numerical behavior for less-tested modes.

## API Design

Add an inplace native entry point:

```text
inplace_dequantize_reduce_mean_update_error_feedback(
    inputs,
    prepared,
    restored,
    residual,
    group_size,
    topk,
    bit,
    quant_type,
    compact,
    divisor,
) -> bool
```

The function returns `true` when the fused CUDA kernel ran and `false` when the caller should use fallback. It writes:

- `restored[index] = reduce_sum(dequantized_inputs[index]) / divisor`
- `residual[index] = prepared[index] - restored[index]`

The caller owns `restored` and `residual` buffers. This lets DDP hook code reuse bucket-level workspace and avoid per-call restored allocation.

## Data Flow

```text
quantized payloads from ranks
        |
        v
fused CUDA kernel
        |
        +--> restored workspace, already mean-reduced when requested
        |
        +--> residual workspace, updated from prepared - restored
```

## Error Handling

The fused function is intentionally capability-gated. It returns `false` instead of throwing for unsupported fast-path predicates. It still throws on invalid tensor contracts that indicate caller bugs, such as mismatched devices, non-contiguous workspace, or dtype mismatch between `prepared`, `restored`, and `residual`.

## Testing

Unit tests should verify:

- pybind export exists
- Python codec wrapper calls the inplace fused symbol with caller-provided output workspace
- unsupported reduce mode is rejected before native call
- CUDA smoke compares fused restored and residual against existing reference operations

Benchmarking remains separate and must be reported as synthetic unless run with real model/data under the ParaScale benchmark rules.
