# 实验运行命令说明

本文档列出生成 `results/`（原始 Qwen 结果 + MiniCPM 结果）和 `results_strict/`（4B/9B 严格协议结果）中每个实验的命令。所有命令都从仓库根目录执行。

**环境**：`conda activate sp`；Python 3.10.22；PyTorch 2.8.0+cu126；Transformers 5.20.2。  
**仓库**：`/data1/suzeqin/SpeculativeProbing`  
**GPU**：示例中的 `CUDA_VISIBLE_DEVICES` 可按空闲 GPU 替换；原始 4B/9B 命令没有固定 GPU，27B 训练最后使用 GPU 4。

---

## 0. 通用变量

| 变量 | 4B | 9B | 27B |
|---|---|---|---|
| `TAG` | `4b` | `9b` | `27b` |
| `MODEL` | `models/Qwen3.5-4B` | `models/Qwen3.5-9B` | `models/Qwen3.5-27B` |
| `MTP` | `models/Qwen3.5-4B-mtp` | `models/Qwen3.5-9B-mtp` | `models/Qwen3.5-27B-mtp` |
| 原始 mid layer | 16 | 16 | 32 |
| 论文 mid layer | 8 | 16 | 20 |
| `TARGET`（VerIH / nemotron） | 645 / 800 | 645 / 800 | 645 / 800 |

> 原 4B/9B 命令文件中的 `MODEL` 是 HF id（`Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-9B`）；如果本地已有 `models/Qwen3.5-4B`、`models/Qwen3.5-9B`，直接用本地路径即可。

```bash
conda activate sp
cd /data1/suzeqin/SpeculativeProbing
```

如果使用绝对路径的 Python：

```bash
PY=/home/suzeqin/miniconda3/envs/sp/bin/python
```

---

## 1. 数据与 MTP 头

只需首次运行：

```bash
python verih/download_data.py
python nemotron_safety/build_classifier_dataset.py

python extract_qwen35_mtp.py --src models/Qwen3.5-4B --dst models/Qwen3.5-4B-mtp
python extract_qwen35_mtp.py --src models/Qwen3.5-9B --dst models/Qwen3.5-9B-mtp
python extract_qwen35_mtp.py --src models/Qwen3.5-27B --dst models/Qwen3.5-27B-mtp
```

---

## 2. 原始结果（`results/`）

### 2.1 4B/9B 原始 hidden-state 缓存（mid layer 16）

来自 `4B和9B的命令.md`。注意：对 VerIH，`load_verih_data()` 会忽略 `--num_train/--num_test`；nemotron 会使用这两个参数。

```bash
for spec in \
  "4b models/Qwen3.5-4B models/Qwen3.5-4B-mtp" \
  "9b models/Qwen3.5-9B models/Qwen3.5-9B-mtp"
do
  set -- $spec
  TAG=$1; MODEL=$2; MTP=$3
  MID=16

  python fast_softprompt/precompute_hidden_states.py \
    --model "${MODEL}" --task verih \
    --output "cache_verih_${TAG}.pt" \
    --num_train 10000 --num_test 645 \
    --max_length 2048 --batch_size 4 --mid_layer "${MID}"

  python fast_softprompt/precompute_hidden_states.py \
    --model "${MODEL}" --task nemotron_safety \
    --output "cache_nemotron_safety_${TAG}.pt" \
    --num_train 10000 --num_test 800 \
    --max_length 2048 --batch_size 4 --mid_layer "${MID}"
done
```

产出：

- `cache_verih_4b.pt` + `cache_verih_4b_mid.pt`
- `cache_nemotron_safety_4b.pt` + `cache_nemotron_safety_4b_mid.pt`
- 9B 同名缓存

### 2.2 4B/9B 原始方法（SP / MLP / MultiMax）

**SP-1 / SP-2 / SP-5**（`results/{task}_{TAG}/sp{1,2,5}`）：

```bash
for spec in \
  "4b models/Qwen3.5-4B-mtp" \
  "9b models/Qwen3.5-9B-mtp"
do
  set -- $spec
  TAG=$1; MTP=$2

  for task in verih nemotron_safety; do
    if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
    for k in 1 2 5; do
      python fast_softprompt/train_softprompt.py \
        --cache "cache_${task}_${TAG}.pt" --mmap_cache \
        --mtp_dir "${MTP}" \
        --output_dir "results/${task}_${TAG}/sp${k}" \
        --soft_prompt_length "${k}" --recursive --classifier_head linear \
        --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
        --dropout 0.2 --target_eval "${TE}"
    done
  done
done
```

**MLP**（`results/{task}_{TAG}/mlp`，原始结果使用代码默认 `hidden_dim=1024`）：

```bash
for spec in "4b" "9b"; do
  TAG=$spec
  for task in verih nemotron_safety; do
    if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
    python fast_softprompt/train_mlp.py \
      --cache "cache_${task}_${TAG}.pt" --mmap_cache \
      --output_dir "results/${task}_${TAG}/mlp" \
      --lr 5e-4 --epochs 10 --batch_size 32 --target_eval "${TE}"
  done
done
```

**MultiMax**（`results/{task}_{TAG}/multimax`）：这是原始 4B/9B 命令，使用**主缓存**，不是论文要求的 `_mid`。严格版本见 §3.3。

```bash
for spec in "4b" "9b"; do
  TAG=$spec
  for task in verih nemotron_safety; do
    if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
    python fast_softprompt/train_multimax.py \
      --cache "cache_${task}_${TAG}.pt" --mmap_cache \
      --output_dir "results/${task}_${TAG}/multimax" \
      --num_heads 10 --mlp_width 100 \
      --lr 5e-4 --epochs 10 --batch_size 32 --target_eval "${TE}"
  done
done
```

### 2.3 27B 原始 hidden-state 缓存（mid layer 32）

来自 `run_27B.sh`。27B 的 `MID_LAYER` 由脚本按 `num_hidden_layers/2 = 64/2 = 32` 计算。

```bash
python fast_softprompt/precompute_hidden_states.py \
  --model models/Qwen3.5-27B --task verih \
  --output cache_verih_27b.pt \
  --num_train 10000 --num_test 645 \
  --max_length 2048 --batch_size 1 --mid_layer 32

python fast_softprompt/precompute_hidden_states.py \
  --model models/Qwen3.5-27B --task nemotron_safety \
  --output cache_nemotron_safety_27b.pt \
  --num_train 10000 --num_test 800 \
  --max_length 2048 --batch_size 1 --mid_layer 32
```

产出：

- `cache_verih_27b.pt` + `cache_verih_27b_mid.pt`（mid layer 32）
- `cache_nemotron_safety_27b.pt` + `cache_nemotron_safety_27b_mid.pt`（mid layer 32）

### 2.4 27B 原始方法

**SP-1 / SP-2 / SP-5**（`results/{task}_27b/sp{1,2,5}`）：

```bash
for task in verih nemotron_safety; do
  if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
  for k in 1 2 5; do
    python fast_softprompt/train_softprompt.py \
      --cache "cache_${task}_27b.pt" --mmap_cache \
      --mtp_dir models/Qwen3.5-27B-mtp \
      --output_dir "results/${task}_27b/sp${k}" \
      --soft_prompt_length "${k}" --recursive --classifier_head linear \
      --lr 3e-4 --min_lr 1e-4 --epochs 10 --batch_size 1 \
      --dropout 0.2 --target_eval "${TE}"
  done
done
```

**MLP**（`results/{task}_27b/mlp`，`hidden_dim=1024`）：

```bash
# VerIH：原始运行
python fast_softprompt/train_mlp.py \
  --cache cache_verih_27b.pt --mmap_cache \
  --output_dir results/verih_27b/mlp \
  --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645

# nemotron：在 train_mlp.py 加入 NaN 保护后重跑
python fast_softprompt/train_mlp.py \
  --cache cache_nemotron_safety_27b.pt --mmap_cache \
  --output_dir results/nemotron_safety_27b/mlp \
  --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 800
```

**MultiMax**（`results/{task}_27b/multimax`）：原始 `run_27B.sh` 曾误用主缓存；下面是用 `_mid` 缓存重跑、最终写入结果的命令。注意 27B 的 mid layer 是 32，论文是 20。

```bash
python fast_softprompt/train_multimax.py \
  --cache cache_verih_27b_mid.pt --mmap_cache \
  --output_dir results/verih_27b/multimax \
  --num_heads 10 --mlp_width 100 \
  --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645

python fast_softprompt/train_multimax.py \
  --cache cache_nemotron_safety_27b_mid.pt --mmap_cache \
  --output_dir results/nemotron_safety_27b/multimax \
  --num_heads 10 --mlp_width 100 \
  --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 800
```

---

## 3. 严格协议结果（`results_strict/`）

严格协议：MultiMax 用论文的 mid layer（4B=8，9B=16）；MLP 用论文的 hidden dim 2048。SP 超参数本来与论文一致，结果为原样复制（§3.4）。

### 3.1 4B 的 mid-layer 8 缓存

9B 的 `_mid` 缓存已经是论文要求的 layer 16，无需重建。4B 原来的 `_mid` 是 layer 16，需要重新提取 layer 8：

```bash
CUDA_VISIBLE_DEVICES=4 python fast_softprompt/precompute_mid_from_cache.py \
  --cache cache_verih_4b.pt --model models/Qwen3.5-4B \
  --mid_layer 8 --output cache_verih_4b_mid8.pt --batch_size 8

CUDA_VISIBLE_DEVICES=5 python fast_softprompt/precompute_mid_from_cache.py \
  --cache cache_nemotron_safety_4b.pt --model models/Qwen3.5-4B \
  --mid_layer 8 --output cache_nemotron_safety_4b_mid8.pt --batch_size 16
```

产出：

- `cache_verih_4b_mid8.pt`（layer 8）
- `cache_nemotron_safety_4b_mid8.pt`（layer 8）

### 3.2 严格 MLP（hidden dim 2048）

```bash
for TAG in 4b 9b; do
  for task in verih nemotron_safety; do
    if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
    python fast_softprompt/train_mlp.py \
      --cache "cache_${task}_${TAG}.pt" --mmap_cache \
      --output_dir "results_strict/${task}_${TAG}/mlp" \
      --hidden_dim 2048 \
      --lr 5e-4 --epochs 10 --batch_size 32 --target_eval "${TE}"
  done
done
```

### 3.3 严格 MultiMax（mid layer 8/16，使用 `_mid`）

**9B（layer 16）**：

```bash
for task in verih nemotron_safety; do
  if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
  python fast_softprompt/train_multimax.py \
    --cache "cache_${task}_9b_mid.pt" --mmap_cache \
    --output_dir "results_strict/${task}_9b/multimax" \
    --num_heads 10 --mlp_width 100 \
    --lr 5e-4 --epochs 10 --batch_size 32 --target_eval "${TE}"
done
```

**4B（layer 8，使用 §3.1 生成的缓存）**：

```bash
for task in verih nemotron_safety; do
  if [ "$task" = "verih" ]; then TE=645; else TE=800; fi
  python fast_softprompt/train_multimax.py \
    --cache "cache_${task}_4b_mid8.pt" --mmap_cache \
    --output_dir "results_strict/${task}_4b/multimax" \
    --num_heads 10 --mlp_width 100 \
    --lr 5e-4 --epochs 10 --batch_size 32 --target_eval "${TE}"
done
```

### 3.4 严格目录中的 SP 结果（复制，不是重跑）

`results_strict/` 里的 SP-1/2/5 与 `results/` 中同名文件字节相同；它们由 §2.2 的 SP 命令生成，这里只是复制：

```bash
for TAG in 4b 9b; do
  for task in verih nemotron_safety; do
    for k in 1 2 5; do
      cp -r "results/${task}_${TAG}/sp${k}" "results_strict/${task}_${TAG}/"
    done
  done
done
```

### 3.5 MiniCPM4.1-8B（Eagle3 头）

前置：

- 基座：`models/MiniCPM4.1-8B`
- Eagle3：`models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16`
- transformers 5.x 需要 `fast_softprompt/minicpm_compat.py`（预计算/训练脚本已自动导入）

预计算（VerIH 和 nemotron 各一次）：

```bash
export BASE=models/MiniCPM4.1-8B
export E3DIR=models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python fast_softprompt/precompute_hidden_states_minicpm.py \
  --task verih --output_prefix cache_verih_mcpm --base_model "$BASE" \
  --num_train 10000 --num_test 1000 --max_length 2048 --batch_size 2

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python fast_softprompt/precompute_hidden_states_minicpm.py \
  --task nemotron_safety --output_prefix cache_nemotron_safety_mcpm --base_model "$BASE" \
  --num_train 10000 --num_test 1000 --max_length 2048 --batch_size 2
```

SP-1/2/5（以 VerIH 为例；nemotron 换 cache 和 `--target_eval 800`）：

```bash
for k in 1 2 5; do
  python fast_softprompt/train_eagle3_minicpm.py \
    --cache cache_verih_mcpm_eagle3.pt --mmap_cache \
    --eagle3_dir "$E3DIR" --base_model "$BASE" \
    --output_dir "results/verih_mcpm/sp${k}" \
    --soft_prompt_length "$k" --epochs 10 --batch_size 1 \
    --dropout 0.2 --target_eval 645
done
```

MLP（hidden dim 2048）：

```bash
python fast_softprompt/train_mlp.py \
  --cache cache_verih_mcpm.pt --mmap_cache \
  --output_dir results/verih_mcpm/mlp --hidden_dim 2048 \
  --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645
```

MultiMax（mid layer 16，必须用 `_mid.pt` 和 `--mmap_cache`）：

```bash
python fast_softprompt/train_multimax.py \
  --cache cache_verih_mcpm_mid.pt --mmap_cache \
  --output_dir results/verih_mcpm/multimax \
  --num_heads 10 --mlp_width 100 \
  --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645
```

实际划分：VerIH 5,705/322/323；nemotron 10,000/440/439（论文为 5,000/300/300 和 10,000/500/500）。CoT 两个任务未跑（缺少标注数据）。

---

## 4. 结果路径 → 生成命令

| 结果路径 | 生成命令 | 备注 |
|---|---|---|
| `results/{verih,nemotron_safety}_{4b,9b}/sp{1,2,5}` | §2.2 SP 循环 | 原始划分 |
| `results/{verih,nemotron_safety}_{4b,9b}/mlp` | §2.2 MLP 循环 | `hidden_dim=1024` |
| `results/{verih,nemotron_safety}_{4b,9b}/multimax` | §2.2 MultiMax 循环 | **主缓存，非严格** |
| `results/{verih,nemotron_safety}_27b/sp{1,2,5}` | §2.4 SP 循环 | 原始划分 |
| `results/verih_27b/mlp` | §2.4 27B MLP（VerIH） | `hidden_dim=1024` |
| `results/nemotron_safety_27b/mlp` | §2.4 27B MLP（nemotron，NaN 保护后重跑） | `hidden_dim=1024` |
| `results/{verih,nemotron_safety}_27b/multimax` | §2.4 27B MultiMax（`_mid` layer 32） | 论文为 layer 20 |
| `results_strict/{verih,nemotron_safety}_{4b,9b}/mlp` | §3.2 | `hidden_dim=2048` |
| `results_strict/{verih,nemotron_safety}_{4b,9b}/multimax` | §3.3 | 4B layer 8，9B layer 16 |
| `results_strict/{verih,nemotron_safety}_{4b,9b}/sp{1,2,5}` | §3.4 复制；原始命令是 §2.2 | 不是重跑 |
| `results/{verih,nemotron_safety}_mcpm/sp{1,2,5}` | §3.5 SP | Eagle3 头 |
| `results/{verih,nemotron_safety}_mcpm/mlp` | §3.5 MLP | `hidden_dim=2048` |
| `results/{verih,nemotron_safety}_mcpm/multimax` | §3.5 MultiMax | mid layer 16，`_mid` 缓存 |

---

## 5. 代码前提与有效性说明

1. **`fast_softprompt/train_softprompt.py` 的 best-state 修复**必须存在，否则 27B 在 Epoch 2 刷新 best 时可能 OOM（原代码在 GPU 上克隆整个冻结 MTP 头）。该修复不影响训练数值。
2. **`fast_softprompt/train_mlp.py` 的非有限 loss 保护**必须存在，否则 27B nemotron 缓存中的 NaN 样本会污染整个 MLP，得到 val 0.5 / F1 0。本报告中的 27B nemotron MLP 结果是在该保护下重跑的。
3. **`run_27B.sh` 的 `MID_CACHE` 修复**：原始脚本给 `train_multimax.py` 传主缓存；最终 27B MultiMax 结果是用 `_mid` 缓存重跑的。
4. **27B 的 MLP/SP 不是论文 27B 的严格复现**：论文 27B hidden size 为 3584，公开 checkpoint 为 5120；相关结果只能作为当前公开模型的参考。
5. **27B MultiMax 的 mid layer 是 32**（论文为 20），因此 27B MultiMax 也不是严格协议。
6. **MiniCPM 依赖 `fast_softprompt/minicpm_compat.py`**：transformers 5.x 移除了 `is_torch_fx_available`，没有该 shim 时 MiniCPM 远程代码会 ImportError。
7. **MiniCPM 数据划分**：VerIH 5,705/322/323；nemotron 10,000/440/439；论文为 5,000/300/300 和 10,000/500/500。
8. **MiniCPM 的 CoT 两个任务未跑**：`cot_inconsistency/data/` 为空，需要先生成/标注数据。
6. **4B/9B 严格结果**已完成 MultiMax layer 8/16 和 MLP hidden 2048；唯一剩余差异是数据划分（论文 5,000/300/300、10,000/500/500；本复现 5,729/322/323、10,000/400/400）。
7. **27B SP-5 VerIH** 曾被后台会话中断一次，由 watchdog 用同一条命令重启；最终结果来自重启后的完整训练。
8. 原始 4B/9B 的 MultiMax 结果使用主缓存，数值接近论文但协议不对；严格结果见 `results_strict/`。

---

## 6. 诊断命令（不产生最终结果）

**27B MLP hidden dim=2048 诊断**：

```bash
python fast_softprompt/train_mlp.py \
  --cache cache_verih_27b.pt --mmap_cache \
  --output_dir /tmp/mlp2048_verih_27b \
  --hidden_dim 2048 --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645
```

**27B OOM 修复的 1-epoch 冒烟测试**：

```bash
CUDA_VISIBLE_DEVICES=4 python fast_softprompt/train_softprompt.py \
  --cache cache_verih_27b.pt --mmap_cache \
  --mtp_dir models/Qwen3.5-27B-mtp \
  --output_dir /tmp/sp_smoke_27b \
  --soft_prompt_length 1 --recursive --classifier_head linear \
  --lr 3e-4 --min_lr 1e-4 --epochs 1 --batch_size 1 \
  --dropout 0.2 --target_eval 645
```

**mid-layer 提取脚本的验证**（用 layer 16 与现有缓存对比）：

```bash
CUDA_VISIBLE_DEVICES=4 python fast_softprompt/precompute_mid_from_cache.py \
  --cache cache_verih_4b.pt --model models/Qwen3.5-4B \
  --mid_layer 16 --output /tmp/test_mid16.pt --batch_size 4 --limit 32
```
