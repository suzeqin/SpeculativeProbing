#!/usr/bin/env bash
set -euo pipefail

# ===== 0. 环境 =====
# conda activate sp

# ===== 1. 本地模型与 MTP 头 =====
if [ ! -d models/Qwen3.5-27B ]; then
  hf download Qwen/Qwen3.5-27B --local-dir ./models/Qwen3.5-27B
fi

if [ ! -d models/Qwen3.5-27B-mtp ]; then
  python extract_qwen35_mtp.py \
    --src models/Qwen3.5-27B \
    --dst models/Qwen3.5-27B-mtp
fi

# ===== 2. 构建数据集 =====
python verih/download_data.py

if [ ! -f nemotron_safety/classifier_dataset.jsonl ]; then
  python nemotron_safety/build_classifier_dataset.py
fi

# ===== 3. 确定 mid_layer =====
# 手动覆盖方式：MID_LAYER_OVERRIDE=32 bash run_27B.sh
if [ -n "${MID_LAYER_OVERRIDE:-}" ]; then
  MID_LAYER="${MID_LAYER_OVERRIDE}"
  echo "[INFO] 使用手动指定的 MID_LAYER=${MID_LAYER}"
else
  NUM_LAYERS=$(python - <<'PY'
import json
cfg = json.load(open('models/Qwen3.5-27B/config.json'))

def find_num_layers(obj):
    if isinstance(obj, dict):
        for key in ('num_hidden_layers', 'n_layer', 'num_layers',
                    'layer_num', 'num_transformer_layers', 'n_layers'):
            v = obj.get(key)
            if isinstance(v, int) and v > 0:
                return v
        for key in ('text_config', 'llm_config', 'model_config',
                    'language_config', 'decoder_config'):
            v = obj.get(key)
            if isinstance(v, dict):
                r = find_num_layers(v)
                if r is not None:
                    return r
        for v in obj.values():
            r = find_num_layers(v)
            if r is not None:
                return r
    return None

v = find_num_layers(cfg)
print(v if v is not None else '')
PY
)

  if [ -z "${NUM_LAYERS}" ]; then
    echo "[ERROR] 无法从 config.json 自动读取层数。"
    echo "请运行以下命令查看 config，然后手动指定："
    echo "  python -c \"import json; print(json.dumps(json.load(open('models/Qwen3.5-27B/config.json')), indent=2))\""
    echo "  MID_LAYER_OVERRIDE=<层数一半> bash run_27B.sh"
    exit 1
  fi

  MID_LAYER=$((NUM_LAYERS / 2))
  echo "[INFO] num_hidden_layers=${NUM_LAYERS}, mid_layer=${MID_LAYER}"
fi

# ===== 4. 预计算隐藏状态缓存 =====
if [ ! -f cache_verih_27b.pt ] || [ ! -f cache_verih_27b_mid.pt ]; then
  python fast_softprompt/precompute_hidden_states.py \
    --model models/Qwen3.5-27B \
    --task verih \
    --output cache_verih_27b.pt \
    --num_train 10000 \
    --num_test 645 \
    --max_length 2048 \
    --batch_size 1 \
    --mid_layer "${MID_LAYER}"
fi

if [ ! -f cache_nemotron_safety_27b.pt ] || [ ! -f cache_nemotron_safety_27b_mid.pt ]; then
  python fast_softprompt/precompute_hidden_states.py \
    --model models/Qwen3.5-27B \
    --task nemotron_safety \
    --output cache_nemotron_safety_27b.pt \
    --num_train 10000 \
    --num_test 800 \
    --max_length 2048 \
    --batch_size 1 \
    --mid_layer "${MID_LAYER}"
fi

# ===== 5. 训练 SP-1 / SP-2 / SP-5 与基线 =====
mkdir -p results

for task in verih nemotron_safety; do
  if [ "$task" = "verih" ]; then
    CACHE=cache_verih_27b.pt
    MID_CACHE=cache_verih_27b_mid.pt
    TARGET=645
  else
    CACHE=cache_nemotron_safety_27b.pt
    MID_CACHE=cache_nemotron_safety_27b_mid.pt
    TARGET=800
  fi

  echo "[INFO] ===== task=${task}, cache=${CACHE}, target_eval=${TARGET} ====="

  for k in 1 2 5; do
    if [ -f "results/${task}_27b/sp${k}/final_results.json" ]; then
      echo "[INFO] skipping SP-${k} on ${task} (already finished)"
      continue
    fi
    echo "[INFO] training SP-${k} on ${task}"
    python fast_softprompt/train_softprompt.py \
      --cache "${CACHE}" --mmap_cache \
      --mtp_dir models/Qwen3.5-27B-mtp \
      --output_dir "results/${task}_27b/sp${k}" \
      --soft_prompt_length "${k}" --recursive --classifier_head linear \
      --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
      --dropout 0.2 --target_eval "${TARGET}"
  done

  if [ -f "results/${task}_27b/mlp/final_results.json" ]; then
    echo "[INFO] skipping MLP probe on ${task} (already finished)"
  else
    echo "[INFO] training MLP probe on ${task}"
    python fast_softprompt/train_mlp.py \
      --cache "${CACHE}" --mmap_cache \
      --output_dir "results/${task}_27b/mlp" \
      --lr 5e-4 --epochs 10 --batch_size 32 \
      --target_eval "${TARGET}"
  fi

  if [ -f "results/${task}_27b/multimax/final_results.json" ]; then
    echo "[INFO] skipping MultiMax probe on ${task} (already finished)"
  else
    echo "[INFO] training MultiMax probe on ${task}"
    # MultiMax reads the mid-layer cache, not the main cache (see README /
    # scripts/run_qwen.sh: "${CACHE%.pt}_mid.pt").
    python fast_softprompt/train_multimax.py \
      --cache "${MID_CACHE}" --mmap_cache \
      --output_dir "results/${task}_27b/multimax" \
      --num_heads 10 --mlp_width 100 \
      --lr 5e-4 --epochs 10 --batch_size 32 \
      --target_eval "${TARGET}"
  fi
done

# ===== 6. 汇总结果 =====
echo "[INFO] ===== final results ====="
for f in results/verih_27b/*/final_results.json results/nemotron_safety_27b/*/final_results.json; do
  echo "== ${f} =="
  cat "${f}"
  echo
done