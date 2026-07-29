# CUDA Completion Integration Design

## Goal

Unify asynchronous completion semantics across CCDL point-to-point,
topology, all-gather, all-reduce, and reduce-scatter paths while preserving
the current public APIs. CUDA communication, post-processing, and consumer
stream visibility must be ordered without adding unconditional CPU
synchronization.

## Constraints

- Preserve existing function signatures and `async_op=True` return behavior.
- Keep synchronous behavior as the safe fallback for CPU tensors, unavailable
  CUDA runtimes, unsupported distributed backends, and injected test
  transports.
- Do not make PyTorch or CUDA a hard import-time dependency.
- A work object owns every tensor needed by an in-flight operation until
  completion.
- `wait()` returns the final operation result.
- `query()` observes completion and must not initiate synchronization or
  deferred computation.
- Performance validation uses identical tensor shapes, compression settings,
  warm-up counts, and iteration counts for synchronous and asynchronous runs.

## Considered Approaches

### 1. Wrap completed collectives

Keep the existing synchronous implementations and return a work wrapper after
the result is already available.

This is the smallest change and preserves compatibility, but it provides no
communication/computation overlap. It is the behavior currently represented
by `ImmediateWork` and is insufficient for the performance goal.

### 2. Wrap every collective at its public API boundary

Move public calls onto a CUDA stream and record a completion event around the
whole implementation.

This centralizes policy, but it cannot reliably order injected transports,
multi-stage topology algorithms, or receive-side dequantization. It also makes
buffer ownership hard to express because the outer wrapper cannot see every
temporary allocation.

### 3. Unified work protocol with transport-level integration

Define one result-bearing work implementation backed by
`CudaCompletionManager`. Each transport starts distributed work on the
communication stream, runs dependent dequantization or reduction on the same
ordered stream, records an event, and returns the unified work object.
Synchronous callers execute the same path and wait before returning.

This approach requires incremental changes in several transports, but it
provides real ordering, explicit buffer ownership, consistent fallback
behavior, and a stable public API. This is the selected approach.

## Architecture

`CollectiveWork` becomes the public behavioral protocol:

- `wait()` waits for the distributed handle and CUDA completion event, then
  returns the result.
- `query()` returns whether both the underlying communication and the CUDA
  event have completed.
- `get_future()` is exposed when the backend supplies a future; otherwise it
  returns `None`.

`CudaCompletionManager` remains the single runtime boundary. It creates a
CUDA-stream-backed work object when CUDA is available and a safe fallback work
object otherwise. The work object retains the operation result, communication
handle, completion callback, and temporary buffers.

Transport implementations own launch order:

1. The caller's current stream produces the source tensor.
2. The communication stream waits for the caller stream.
3. Quantization and the distributed operation are launched in order.
4. Receive-side dequantization, reduction, mean scaling, and error-feedback
   updates are launched after the distributed handle is complete.
5. A CUDA event is recorded after post-processing.
6. The caller stream waits for that event before consuming the returned
   tensor.

No Python callback may access a receive buffer before its distributed handle
has completed.

## Integration Order

### Point-to-point

`iqsend` and `iqrecv` return unified work objects. Send work retains the
quantized buffer until completion. Receive work retains the compressed buffer
and schedules dequantization once the receive handle completes. Blocking
`qsend` and `qrecv` reuse the same internal path and call `wait()`.

### Topology transports

Replace `_TopologyWork` and `_CallbackTopologyWork` with the shared work
semantics. Existing tree, ring, p2p, and overlap algorithms remain unchanged
initially; only launch ordering, callback execution, result ownership, and
completion reporting are unified.

### All-gather and all-reduce

Add asynchronous torch transports for compressed payloads. Public
`compressed_all_gather` and `compressed_all_reduce` use them when
`async_op=True`. Injected synchronous transports retain `ImmediateWork`
fallback semantics so custom backends remain compatible.

### Reduce-scatter

Reuse the existing async shard pipeline and completion manager, then align its
returned work object with the common protocol. The reduced shard and workspace
buffers remain owned until the completion event is visible.

## Error Handling

- Backend launch errors propagate immediately.
- Backend errors raised by `wait()` propagate from unified `wait()`.
- A post-processing callback runs at most once.
- Failed callbacks are cached and re-raised on subsequent `wait()` calls.
- `query()` returns `False` when the backend cannot report readiness.
- Unsupported CUDA capabilities select the synchronous fallback rather than
  silently returning an incomplete result.

## Testing

Unit tests use fake distributed handles, streams, and events to prove:

- communication handle completion precedes dequantization;
- post-processing runs exactly once;
- `query()` has no side effects;
- temporary buffers remain reachable until completion;
- CPU and missing-CUDA fallbacks preserve results;
- existing public return values remain compatible.

Distributed A6000 tests cover two and four ranks for:

- point-to-point send/receive;
- topology overlap methods;
- compressed all-gather and all-reduce;
- compressed reduce-scatter shard.

Each benchmark records synchronous and asynchronous latency, throughput,
overlap effectiveness, relative L2 error, maximum absolute error, CUDA/PyTorch
versions, GPU topology, tensor size, bit width, group size, warm-up count, and
measured iteration count. Asynchronous timing includes the final `wait()` and
uses a controlled compute workload between launch and wait to measure useful
overlap.

## Acceptance Criteria

- All existing tests pass.
- Every asynchronous path returns an object supporting `wait()` and `query()`.
- Results match the synchronous path within the configured quantization error.
- No unconditional `torch.cuda.synchronize()` is added to library code.
- Work objects retain all in-flight buffers until completion.
- A6000 two-rank and four-rank reports show both raw async overhead and overlap
  benefit using the same benchmark configuration.
