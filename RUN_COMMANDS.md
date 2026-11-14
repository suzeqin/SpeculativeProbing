# 实验运行命令说明

本文档列出生成 `results/` 中每个实验的命令（Qwen 4B/9B 严格结果 + MiniCPM 结果；原始/27B 命令保留作重建参考）。所有命令都从仓库根目录执行。

> **清理说明（2026-11-09）**：已删除 27B 全部产物、原始 4B/9B 非严格结果和未使用的 4B layer-16 mid 缓存。本文档中的原始/27B 命令保留作为重建参考；最终表格对应的命令见 §3.2（严格 MLP）、§3.3（严格 MultiMax）和 §3.5（MiniCPM）。

> **一键复现**：`bash run_all_repro.sh --gpus 0,5,6,3`。默认把 30 个结果写入 `results2/`，并与 `results/` 自动校验（默认容差 3 pp）。

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

来自原始 4B/9B 复现流程（历史命令文件已删除；命令保留如下）。注意：对 VerIH，`load_verih_data()` 会忽略 `--num_train/--num_test`；nemotron 会使用这两个参数。

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

来自 27B 复现流程（原脚本已删除；命令保留如下）。27B 的 `MID_LAYER` 原按 `num_hidden_layers/2 = 64/2 = 32` 计算。

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

**MultiMax**（`results/{task}_27b/multimax`）：原始 27B 流程曾误用主缓存；下面是用 `_mid` 缓存重跑、最终写入结果的命令。注意 27B 的 mid layer 是 32，论文是 20。

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

## 3. 严格协议结果（`results/`）

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
      --output_dir "results/${task}_${TAG}/mlp" \
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
    --output_dir "results/${task}_9b/multimax" \
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
    --output_dir "results/${task}_4b/multimax" \
    --num_heads 10 --mlp_width 100 \
    --lr 5e-4 --epochs 10 --batch_size 32 --target_eval "${TE}"
done
```

### 3.4 严格结果中的 SP 结果（原样复制，不是重跑）

严格结果中的 SP-1/2/5 沿用原始 SP 命令（§2.2）的结果，超参与论文一致，没有重跑；它们现在与 MLP/MultiMax 一起位于 `results/{task}_{TAG}/sp{k}`。

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
| `results/{verih,nemotron_safety}_{4b,9b}/sp{1,2,5}` | §3.4（原始命令 §2.2） | 严格协议，SP 原样沿用 |
| `results/{verih,nemotron_safety}_{4b,9b}/mlp` | §3.2 | `hidden_dim=2048` |
| `results/{verih,nemotron_safety}_{4b,9b}/multimax` | §3.3 | 4B layer 8，9B layer 16，`_mid` |
| `results/{verih,nemotron_safety}_mcpm/sp{1,2,5}` | §3.5 SP | Eagle3 头 |
| `results/{verih,nemotron_safety}_mcpm/mlp` | §3.5 MLP | `hidden_dim=2048` |
| `results/{verih,nemotron_safety}_mcpm/multimax` | §3.5 MultiMax | mid layer 16，`_mid` 缓存 |

> §2.1–§2.4 的原始/27B 命令保留作为重建参考；对应的原始 4B/9B、27B 结果已删除。

---

## 5. 代码前提与有效性说明

1. **`fast_softprompt/train_softprompt.py` 的 best-state 修复**必须存在，否则 27B 在 Epoch 2 刷新 best 时可能 OOM（原代码在 GPU 上克隆整个冻结 MTP 头）。该修复不影响训练数值。
2. **`fast_softprompt/train_mlp.py` 的非有限 loss 保护**必须存在，否则 27B nemotron 缓存中的 NaN 样本会污染整个 MLP，得到 val 0.5 / F1 0。本报告中的 27B nemotron MLP 结果是在该保护下重跑的。
3. **原 27B 脚本的 `MID_CACHE` 修复（脚本已删除）**：原始脚本给 `train_multimax.py` 传主缓存；最终 27B MultiMax 结果是用 `_mid` 缓存重跑的。
4. **27B 的 MLP/SP 不是论文 27B 的严格复现**：论文 27B hidden size 为 3584，公开 checkpoint 为 5120；相关结果只能作为当前公开模型的参考。
5. **27B MultiMax 的 mid layer 是 32**（论文为 20），因此 27B MultiMax 也不是严格协议。
6. **MiniCPM 依赖 `fast_softprompt/minicpm_compat.py`**：transformers 5.x 移除了 `is_torch_fx_available`，没有该 shim 时 MiniCPM 远程代码会 ImportError。
7. **MiniCPM 数据划分**：VerIH 5,705/322/323；nemotron 10,000/440/439；论文为 5,000/300/300 和 10,000/500/500。
8. **MiniCPM 的 CoT 两个任务未跑**：`cot_inconsistency/data/` 为空，需要先生成/标注数据。
6. **4B/9B 严格结果**已完成 MultiMax layer 8/16 和 MLP hidden 2048；唯一剩余差异是数据划分（论文 5,000/300/300、10,000/500/500；本复现 5,729/322/323、10,000/400/400）。
7. **27B SP-5 VerIH** 曾被后台会话中断一次，由 watchdog 用同一条命令重启；最终结果来自重启后的完整训练。
8. 原始 4B/9B 的 MultiMax 结果曾使用主缓存（该结果已删除）；严格结果见 `results/`。

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

---

## 7. 推理时间开销测试（新增，`latency/`）

论文正文（§3.3 只给 `O(k n d_h)+O(k d²)` 的渐近项）、Table 6（渐近复杂度）和附录 B（只给训练硬件与训练耗时）都没有报告 SP-1 / SP-2 / SP-5、MLP、MultiMax 的**实测推理延迟**。本节脚本在同一套复现检查点与数据上补齐这项测量，结果见仓库根目录的 [`LATENCY_REPORT.md`](LATENCY_REPORT.md)。

### 7.1 测量内容

- `base_prefill`：基座模型对 prompt 的前向（`output_hidden_states=True, use_cache=True`），即所有 probe 消费特征的同一趟前向；
- `base_decode`：带 KV cache 的单 token 解码步（监控生成过程时的参考成本）；
- `mlp` / `mm`：复现代码中的 `train_mlp.MLPClassifier` / `train_multimax.MultiMaxClassifier`；
- `sp{k}`：**standalone**，即复现代码原样的整序列 draft-head 前向 + k 个 soft prompt；
- `sp{k}_inc`：**KV-cached**，即论文 §3.3 的部署方式：draft head 的 prefix KV cache 已存在，只计算 k 个 soft-prompt 位置（k 个单 token draft-head step + 线性头）；与参考前向的 logits 等价（bf16 下 |Δlogit| ≤ 0.05）；
- `sp{k}_core`（仅 MiniCPM）：去掉分类器从不读取的 draft `lm_head`+`d2t` 投影后的版本。

计时方式：batch size 1、bf16、PyTorch eager；每个样本 5 次 CUDA event 计时（每次之间 flush 32 MiB L2），取每样本最小值再取测试集中位数；同时用 `torch.profiler` 记录单次调用的 CUDA kernel 时间。基座前向与 probe 在同一时间窗口内交错测量，因此**比值**不受 GPU 争用影响（绝对值会受影响，见报告 §8）。

### 7.2 运行命令

```bash
conda activate sp
cd /data1/suzeqin/SpeculativeProbing

# 单个 (model, task)
CUDA_VISIBLE_DEVICES=6 python latency/bench_latency.py \
  --model qwen3.5-4b --task verih \
  --methods mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc \
  --reps 5 --reps_base 2 --profile_reps 5 \
  --out latency/results/qwen3.5-4b_verih.json

# 全部 6 个 (model, task)，约 40 分钟（共享 A6000）
bash latency/run_all_latency.sh 6

# 只测 probe 的随长度缩放（不需要基座模型，约 30 秒/配置）
bash latency/run_scaling.sh 6

# 由 JSON 重新生成 CSV / Markdown 表
python latency/merge_latency.py
```

MiniCPM 的两个任务需要额外的 `sp{k}_core`（脚本已包含）：

```bash
python latency/bench_latency.py \
  --model minicpm4.1-8b --task verih \
  --methods mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc,sp1_core,sp2_core,sp5_core \
  --out latency/results/minicpm4.1-8b_verih.json
```

### 7.3 主要结果（batch 1，中位长度样本，单位 ms）

| 模型 | 任务 | prefill | MLP | MultiMax | SP-1 | SP-5 | SP-1 KV | SP-5 KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 60.3 | 0.034 | 0.068 | 3.18 | 24.1 | 0.68 | 5.59 |
| Qwen3.5-4B | nemotron | 47.8 | 0.034 | 0.048 | 3.03 | 24.0 | 0.67 | 5.59 |
| Qwen3.5-9B | verih | 76.6 | 0.044 | 0.078 | 3.58 | 25.9 | 1.01 | 9.58 |
| Qwen3.5-9B | nemotron | 41.1 | 0.043 | 0.11 | 1.60 | 7.26 | 1.13 | 5.03 |
| MiniCPM4.1-8B | verih | 108.0 | 0.044 | 0.080 | 6.38 | 6.41 | 1.08 | 9.93 |
| MiniCPM4.1-8B | nemotron | 101.2 | 0.043 | 0.058 | 6.17 | 6.21 | 1.07 | 9.91 |

结论（详见 `LATENCY_REPORT.md`）：

1. **MLP / MultiMax 基本免费**：0.03–0.11 ms，占 prefill 的 0.03–0.22%；MultiMax 随序列长度缓慢增长（50 token 时约 50 µs，500 token 时约 100 µs）。
2. **SP-1 在 KV-cached 部署下约 0.7–1.1 ms**（占 prefill 0.8–2.4%），即一个 draft-head 解码步——论文“negligible overhead”的说法在这一配置下成立。
3. **SP-5 在 KV-cached 部署下约 5–10 ms**（占 prefill 6.6–10.8%）；成本随 k 近似线性，GPU 时间每步约 0.6 ms（Qwen-4B）、约 1.0 ms（Qwen-9B / MiniCPM Eagle3）。
4. **standalone 路径远慢于渐近分析**：Qwen 的 SP-5 达到 24–26 ms。原因不是算力，而是复现版 `RecursiveSoftPromptClassifier` 在 k>1 的每个递归步里都在 host 上新建 `position_ids` / `cache_position` 张量并做 pageable H2D 拷贝（单次约 2.3 ms）。`latency/bench_latency.py` 的增量实现把这些张量提到循环外，同一算术只需 5–10 ms；把该修复用于参考实现可去掉约 18–20 ms 的 SP-5 延迟。
5. **MiniCPM Eagle3 probe 最贵**：每步 0.7–1.0 ms（SP-1），k≥2 时 eager 约 2 ms/步；其 standalone 前向还有约 2.7–2.9 ms 花在分类器不使用的 draft `lm_head`+`d2t` 上，去掉后从 6.4 ms 降到 3.6 ms。
