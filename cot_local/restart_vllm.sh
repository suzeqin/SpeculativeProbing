#!/bin/bash
# Restart the 122B teacher judge for labeling (see cot_local/label_traces.py).
# The Qwen3.5 chat template runs with thinking ON by default; label_traces.py
# accepts only the verdict written after </think>. No --reasoning-parser and no
# chat_template_kwargs are needed.
export PATH=$HOME/miniconda3/envs/serve27/bin:$PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /data1/suzeqin/SpeculativeProbing
# A 16384 context gives the judge enough room to finish its <think> reasoning
# before emitting </think> + verdict; label_traces.py uses context=16384 to
# match. Keep the two in sync (MAX_MODEL_LEN overrides for debugging only).
exec vllm serve models/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --port 8000 \
  --served-model-name Qwen3.5-122B-A10B-GPTQ-Int4 \
  --tensor-parallel-size 4 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization 0.97 \
  --skip-mm-profiling
