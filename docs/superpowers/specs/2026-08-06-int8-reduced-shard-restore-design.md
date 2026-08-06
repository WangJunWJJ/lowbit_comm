# INT8 ReducedShard Restore Design

## Goal

Reduce the final full-gradient restoration traffic of the compressed
reduce-scatter DDP path by gathering quantized ReducedShard payloads and
dequantizing them into the full-precision DDP bucket only after communication.

## Interface

`make_torch_compressed_reduce_scatter_all_gather()` gains an explicit
`restore_mode` argument. `"fp16"` preserves the existing implementation and
remains the default. `"compressed"` selects the new quantized restoration
path. Unsupported modes fail at construction time.

The transport accepts injectable restore quantize/dequantize functions and an
optional compressed-gather workspace allocator. These interfaces keep unit
tests deterministic and allow the CUDA compiler to bind pooled workspaces
without coupling transport selection to `CompressionConfig`.

## Data Flow

1. The existing compressed reduce-scatter produces one full-precision
   `ReducedShard` per rank.
2. Compressed restore quantizes that shard using the active compression policy.
3. Ranks all-gather equal-sized packed byte payloads. The packed payload
   already contains the group scales required by the CCDL codec.
4. Each gathered payload is dequantized directly into its corresponding slice
   of the final full-precision output workspace.
5. Padding is trimmed and the original bucket shape is restored before the DDP
   future completes.

All ranks execute the same deterministic order and receive the same payloads,
so they reconstruct identical approximate gradients. Standard DDP and AdamW
semantics remain unchanged.

## Safety and Fallback

The existing full-precision restoration remains available as an explicit
fallback. Compressed restore validates the gathered workspace capacity and
requires tensor collectives; invalid contracts fail before returning a bucket
to DDP. Runtime benchmark configuration may fall back to `"fp16"` when the
CUDA extension is unavailable.

## Verification

Unit tests prove that the collective transports quantized payloads rather than
FP shards, that every payload is decoded into the correct final slice, and that
padding and asynchronous completion preserve current behavior. CUDA smoke
tests compare reconstructed tensors against codec output. A6000 validation
uses the existing 21 GB PSI Policy workload with identical model, batch size,
GPU placement, epoch count, and native-DDP baseline for 2 and 4 GPUs.

