# Quantized point-to-point smoke validation, 2026-07-29

This report records the migration of legacy CCDL `qsend/qrecv/iqsend/iqrecv`
into the refactored `ccdl_comm` package.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`

## Result

| GPUs | Numel | Dtype | Bit | Group size | Blocking relative L2 | Async relative L2 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2 | 1,048,576 | fp16 | 8 | 64 | 0.00594 | 0.00594 |

Both blocking and async quantized P2P paths complete on A6000/NCCL and produce
the expected INT8 quantization error envelope.
