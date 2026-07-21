#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${DATA_PATH:?DATA_PATH must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
MAX_STEPS="${MAX_STEPS:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_BATCHES="${EVAL_BATCHES:-25}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
read -r -a variants <<< "${VARIANTS:-nccl_fp32 int8_k0 int8_k2 int4_k0 int4_k2}"
read -r -a seeds <<< "${SEEDS:-17 29 43}"
mkdir -p "$OUTPUT_ROOT"

for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    run_dir="$OUTPUT_ROOT/$variant-seed$seed"
    [[ -s "$run_dir/COMPLETED.json" ]] && continue
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 -m benchmarks.lm.train \
      --variant "$variant" --seed "$seed" --model-path "$MODEL_PATH" --data-path "$DATA_PATH" \
      --output-dir "$run_dir" --max-steps "$MAX_STEPS" --eval-interval "$EVAL_INTERVAL" \
      --eval-batches "$EVAL_BATCHES" --max-length "$MAX_LENGTH" \
      --micro-batch-size "$MICRO_BATCH_SIZE" \
      --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  done
done
