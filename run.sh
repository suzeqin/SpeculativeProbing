python fast_softprompt/train_softprompt.py \
  --cache cache_verih_9b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-9B-mtp \
  --output_dir results/verih_9b/sp1 \
  --soft_prompt_length 1 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
  --dropout 0.2 --target_eval 645
python fast_softprompt/train_softprompt.py \
  --cache cache_verih_9b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-9B-mtp \
  --output_dir results/verih_9b/sp2 \
  --soft_prompt_length 2 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
  --dropout 0.2 --target_eval 645
python fast_softprompt/train_softprompt.py \
  --cache cache_verih_9b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-9B-mtp \
  --output_dir results/verih_9b/sp5 \
  --soft_prompt_length 5 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
  --dropout 0.2 --target_eval 645
python fast_softprompt/train_mlp.py \
  --cache cache_verih_9b.pt --mmap_cache \
  --output_dir results/verih_9b/mlp \
  --lr 5e-4 --epochs 10 --batch_size 32 \
  --target_eval 645
python fast_softprompt/train_multimax.py \
  --cache cache_verih_9b.pt --mmap_cache \
  --output_dir results/verih_9b/multimax \
  --num_heads 10 --mlp_width 100 \
  --lr 5e-4 --epochs 10 --batch_size 32 \
  --target_eval 645
python fast_softprompt/train_softprompt.py \
  --cache cache_nemotron_safety_9b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-9B-mtp \
  --output_dir results/nemotron_safety_9b/sp1 \
  --soft_prompt_length 1 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
  --dropout 0.2 --target_eval 800
python fast_softprompt/train_softprompt.py \
  --cache cache_nemotron_safety_9b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-9B-mtp \
  --output_dir results/nemotron_safety_9b/sp2 \
  --soft_prompt_length 2 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
  --dropout 0.2 --target_eval 800
python fast_softprompt/train_softprompt.py \
  --cache cache_nemotron_safety_9b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-9B-mtp \
  --output_dir results/nemotron_safety_9b/sp5 \
  --soft_prompt_length 5 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
  --dropout 0.2 --target_eval 800
python fast_softprompt/train_mlp.py \
  --cache cache_nemotron_safety_9b.pt --mmap_cache \
  --output_dir results/nemotron_safety_9b/mlp \
  --lr 5e-4 --epochs 10 --batch_size 32 \
  --target_eval 800
python fast_softprompt/train_multimax.py \
  --cache cache_nemotron_safety_9b.pt --mmap_cache \
  --output_dir results/nemotron_safety_9b/multimax \
  --num_heads 10 --mlp_width 100 \
  --lr 5e-4 --epochs 10 --batch_size 32 \
  --target_eval 800
