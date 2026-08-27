#!/bin/bash
# End-to-end example for MiniCPM4.1-8B (post-hoc-distilled Eagle3 head) on ONE task.
#
# Usage:  bash scripts/run_minicpm.sh <task>
#   <task> in {verih, nemotron_safety, cot_repetitive, branched_thinking}
#
# Run from the repository root. Downloads openbmb/MiniCPM4.1-8B and
# openbmb/MiniCPM4.1-8B-Eagle3 from HuggingFace on first use.
set -euo pipefail

TASK=${1:?usage: run_minicpm.sh <task>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Path to the extracted Eagle3 head directory inside the HF snapshot.
E3DIR=${E3DIR:?set E3DIR to the local MiniCPM4_1-8B-Eagle3-bf16 directory (see README step 1)}

PREFIX=cache_${TASK}_mcpm
OUT=results/${TASK}_mcpm

# --- 1. Precompute (writes PREFIX.pt, PREFIX_mid.pt, PREFIX_eagle3.pt) ---
case "$TASK" in
  verih|nemotron_safety)
    python fast_softprompt/precompute_hidden_states_minicpm.py \
      --task "$TASK" --output_prefix "$PREFIX" \
      --num_train 10000 --num_test 1000 --max_length 2048 --batch_size 2
    EPOCHS=10; TE=$( [ "$TASK" = verih ] && echo 645 || echo 800 ) ;;
  cot_repetitive|branched_thinking)
    LABEL=$( [ "$TASK" = cot_repetitive ] && echo repetitive_thinking || echo branched_thinking )
    CTASK=$(  [ "$TASK" = cot_repetitive ] && echo cot_repetitive     || echo cot_branched )
    python fast_softprompt/precompute_hidden_states_minicpm.py \
      --task "$CTASK" \
      --data cot_inconsistency/data/generated_webinstruct_verified_train_qwen3_5_plus_labeled_${LABEL}.jsonl \
      --output_prefix "$PREFIX" \
      --num_train 5000 --num_test 400 --max_length 8192 --batch_size 2
    EPOCHS=4; TE=400 ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

# --- 2. Our method: soft-prompt classifier on the Eagle3 head (SP-1/2/5) ---
for K in 1 2 5; do
  python fast_softprompt/train_eagle3_minicpm.py \
    --cache "${PREFIX}_eagle3.pt" --mmap_cache --eagle3_dir "$E3DIR" \
    --output_dir "$OUT/sp${K}" \
    --soft_prompt_length "$K" --epochs "$EPOCHS" --batch_size 1 --dropout 0.2 --target_eval "$TE"
done

# --- 3. Baselines ---
python fast_softprompt/train_mlp.py \
  --cache "${PREFIX}.pt" --mmap_cache --output_dir "$OUT/mlp" \
  --epochs 10 --batch_size 32 --lr 5e-4 --target_eval "$TE"

python fast_softprompt/train_multimax.py \
  --cache "${PREFIX}_mid.pt" --output_dir "$OUT/multimax" \
  --num_heads 10 --mlp_width 100 --epochs 10 --batch_size 32 --lr 5e-4 --target_eval "$TE"

echo "Done. Test accuracy is in $OUT/*/final_results.json"
