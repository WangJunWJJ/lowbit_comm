#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
mkdir -p "$OUTPUT_ROOT"
status="$OUTPUT_ROOT/matrix-status.jsonl"
variants=(nccl_fp32 ccdl_int8_k0 ccdl_int8_k2 ccdl_int4_k0 ccdl_int4_k2)
seeds=(1337 2027 4099)
epochs="${EPOCHS:-200}"

for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    run_dir="$OUTPUT_ROOT/$variant-seed$seed"
    if [[ -s "$run_dir/complete.json" ]]; then
      continue
    fi
    printf '{"event":"start","variant":"%s","seed":%s,"time":"%s"}\n' "$variant" "$seed" "$(date -Iseconds)" >> "$status"
    if torchrun --standalone --nproc-per-node=2 -m benchmarks.cifar10.train \
      --variant "$variant" --seed "$seed" --epochs "$epochs" \
      --data-root "$DATA_ROOT" --output-dir "$run_dir" --resume; then
      printf '{"event":"success","variant":"%s","seed":%s,"time":"%s"}\n' "$variant" "$seed" "$(date -Iseconds)" >> "$status"
    else
      code=$?
      printf '{"event":"failure","variant":"%s","seed":%s,"exit_code":%s,"time":"%s"}\n' "$variant" "$seed" "$code" "$(date -Iseconds)" >> "$status"
      exit "$code"
    fi
  done
done
