#!/bin/bash
# Restart the 122B teacher for re-labeling. Qwen3.5 chat template honours
# enable_thinking=false (sent per request via chat_template_kwargs), so no
# server-side template override is needed.
export PATH=$HOME/miniconda3/envs/serve27/bin:$PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /data1/suzeqin/SpeculativeProbing
exec vllm serve models/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --port 8000 \
  --served-model-name Qwen3.5-122B-A10B-GPTQ-Int4 \
  --tensor-parallel-size 4 \
  --max-model-len 9216 \
  --gpu-memory-utilization 0.97 \
  --skip-mm-profiling
