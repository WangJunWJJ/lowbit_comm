# DDP Workspace Cache Design

## Goal

Reduce allocation overhead and peak-memory churn in the fused async error-feedback DDP path by reusing per-bucket dequantized output workspace.

## Background

The fused CUDA kernel introduced in `4151e7a` writes both:

- the restored dequantized/reduced tensor;
- the error-feedback residual.

However, the DDP hook still allocates a restored workspace inside every async completion callback. This makes the fused path faster than the safe EF path, but still increases peak memory relative to non-fused EF.

## Design

Add a small Python-side workspace cache owned by the DDP hook closure:

```text
bucket key -> restored workspace tensor
```

The cache key includes:

- bucket key;
- original shape;
- dtype;
- device;
- padded allocation size implied by `CompressionConfig.group_size`.

If a compatible workspace already exists, reuse it. If the bucket shape, dtype, device, or padded size changes, allocate a replacement.

## API

Add:

```text
DequantizedWorkspaceCache
  .get(key, tensor, shape, config) -> tensor
  .clear() -> None
```

The cache uses the existing `allocate_dequantized_buffer` allocator by default and accepts an injected allocator for tests.

## DDP Hook Integration

The DDP hook should:

1. create one cache per hook instance;
2. request workspace with the current bucket key;
3. pass that workspace into `inplace_dequantize_reduce_mean_update_error_feedback`;
4. keep existing fallback paths unchanged.

The cache is only used for the fused async EF path. Non-EF, sync EF, and unsupported native paths remain unchanged.

## Safety

The workspace is reused only within the same DDP hook instance. DDP buckets are processed in deterministic bucket order, and the async pipeline records completion before setting the Future result. Future work should replace CPU synchronization with CUDA event ownership before allowing more aggressive overlap.

## Testing

- Unit-test cache hit/miss behavior with fake tensors.
- Unit-test DDP hook calls the allocator only once across two calls for the same bucket.
- Keep existing fallback tests unchanged.
- Run CUDA smoke tests remotely to verify the extension still builds and fused path remains available.
