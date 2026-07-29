# Dynamic compressed all-gather smoke validation, 2026-07-29

This report records the migration of legacy CCDL `qall_gather_dyn` semantics
into `compressed_all_gather_dynamic` plus the compatibility alias
`qall_gather_dyn`.

## Result

| GPUs | Shapes | Dtype | Bit | Group size | Max relative L2 |
| ---: | --- | --- | ---: | ---: | ---: |
| 2 | `[524288]`, `[524352]` | fp16 | 8 | 64 | 0.00595 |

The implementation exchanges dynamic shape/payload metadata with
`all_gather_object`, pads compressed GPU payloads to the maximum rank-local
compressed length, gathers them with GPU all-gather, trims each payload, and
dequantizes back to the original rank-local shape.
