#!/bin/bash
# End-to-end example for ONE Qwen3.5 model on ONE task.
# Reproduces one row/cell of the paper tables (co-pretrained MTP head).
#
# Usage:  bash scripts/run_qwen.sh <size> <task>
#   <size> in {4B, 9B, 27B}
#   <task> in {verih, nemotron_safety, cot_repetitive, branched_thinking}
#
# Run from the repository root. Assumes the base model has been downloaded and
# its MTP head extracted into models/Qwen3.5-<size>-mtp (see README step 1).
set -euo pipefail

SIZE=${1:?usage: run_qwen.sh <4B|9B|27B> <task>}
TASK=${2:?usage: run_qwen.sh <4B|9B|27B> <task>}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BASE=Qwen/Qwen3.5-${SIZE}          # HF id, or a local path
MTP=models/Qwen3.5-${SIZE}-mtp     # extracted MTP head (README step 1)
CACHE=cache_${TASK}_${SIZE}.pt
OUT=results/${TASK}_${SIZE}

# --- 1. Precompute base-model hidden states (writes CACHE + CACHE.hidden + *_mid.pt) ---
case "$TASK" in
  verih|nemotron_safety)
    python fast_softprompt/precompute_hidden_states.py \
      --model "$BASE" --task "$TASK" --output "$CACHE" \
      --num_train 10000 --num_test 1000 --max_length 2048 --batch_size 4 --mid_layer 16
    EPOCHS=10; TE=$( [ "$TASK" = verih ] && echo 645 || echo 800 ) ;;
  cot_repetitive|branched_thinking)
    # cot_repetitive -> ..._repetitive_thinking.jsonl ; branched_thinking -> ..._branched_thinking.jsonl
    LABEL=$( [ "$TASK" = cot_repetitive ] && echo repetitive_thinking || echo branched_thinking )
    CTASK=$(  [ "$TASK" = cot_repetitive ] && echo cot_repetitive     || echo cot_branched )
    python fast_softprompt/precompute_hidden_states_cot.py \
      --model "$BASE" --task "$CTASK" \
      --data cot_inconsistency/data/generated_webinstruct_verified_train_qwen3_5_plus_labeled_${LABEL}.jsonl \
      --output "$CACHE" \
      --num_train 5000 --num_test 400 --max_length 8192 --batch_size 2 --mid_layer 16
    EPOCHS=4; TE=400 ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

# --- 2. Our method: recursive soft-prompt classifier on the MTP head (SP-1/2/5) ---
for K in 1 2 5; do
  python fast_softprompt/train_softprompt.py \
    --cache "$CACHE" --mmap_cache --mtp_dir "$MTP" \
    --output_dir "$OUT/sp${K}" \
    --soft_prompt_length "$K" --recursive --classifier_head linear \
    --lr 3e-4 --min_lr 1e-4 --epochs "$EPOCHS" --batch_size 1 --dropout 0.2 --target_eval "$TE"
done

# --- 3. Baselines: MLP probe and MultiMax probe (MultiMax reads the *_mid cache) ---
python fast_softprompt/train_mlp.py \
  --cache "$CACHE" --output_dir "$OUT/mlp" \
  --epochs 10 --batch_size 32 --lr 5e-4 --target_eval "$TE"

python fast_softprompt/train_multimax.py \
  --cache "${CACHE%.pt}_mid.pt" --output_dir "$OUT/multimax" \
  --num_heads 10 --mlp_width 100 --epochs 10 --batch_size 32 --lr 5e-4 --target_eval "$TE"

echo "Done. Test accuracy is in $OUT/*/final_results.json"
