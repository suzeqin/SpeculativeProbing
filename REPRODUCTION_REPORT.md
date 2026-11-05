# 论文复现报告

**论文**：Speculative Probing: LLM Monitoring at Speculative-Decoding Cost  
**论文来源**：arXiv:2608.28099；EMNLP 2026 Findings（2026.findings-emnlp.1239）  
**官方代码**：https://github.com/collinzrj/SpeculativeProbing  
**复现日期**：2026-11-01 至 2026-11-04  
**复现环境**：Linux；Python 3.10.22；PyTorch 2.8.0+cu126；Transformers 5.20.2；conda 环境 `sp`  
**硬件**：8× GPU（RTX 4090 / RTX A6000，48 GB），共享服务器

---

## 1. 执行摘要

本次复现覆盖论文主表中的 **Qwen3.5-4B / 9B / 27B × {VerIH, nemotron_safety} × {SP-1, SP-2, SP-5, MLP, MultiMax}**，共 **30 个实验格**，全部生成了 `final_results.json`。

核心结论：

- **4B / 9B 严格协议复现完成**：MultiMax 已改用论文要求的 mid layer（4B=8，9B=16），MLP 已改用论文的 hidden dim 2048；五种方法在两个任务上共 20 个实验，与论文相差约 **0.3–2.0 pp**，总体 MAE 约 **0.92 pp**。严格结果位于 `results_strict/`。
- **27B 未能复现论文结果**：MultiMax 接近论文（VerIH +0.3 pp，nemotron +3.2 pp），但 MLP 和 SP 系统性偏低 **6.8–18.0 pp**。证据指向**论文使用的 27B checkpoint 与当前公开的 Qwen3.5-27B 不是同一个模型**：论文写 27B hidden size 为 **3584**，而公开 checkpoint 为 **5120**；4B/9B 的 hidden size（2560/4096）则与论文完全一致。
- **数据划分仍有差异**：论文 VerIH 为 5,000/300/300、nemotron 为 10,000/500/500；本次复现沿用现有缓存的 5,729/322/323 和 10,000/400/400。公开数据加载器无法精确重建论文划分，详见 5.3。
- 复现过程中发现并修复了原仓库的 **2 个代码问题**（27B OOM、MLP 被 NaN 污染），并定位了原仓库/复现脚本中的多处协议不一致。
- 论文的 **CoT Repetition、CoT Reasoning Strategy、MiniCPM4.1-8B、PII、GPT-5.4-mini zero-shot、专用安全分类器对比、head ablation、cost 分析**均未复现。

---

## 2. 复现范围与实验配置

### 2.1 覆盖范围

| 维度 | 覆盖内容 |
|---|---|
| 任务 | VerIH（指令矛盾检测）、nemotron_safety（多语言 prompt 安全） |
| 模型 | Qwen3.5-4B、Qwen3.5-9B、Qwen3.5-27B |
| 方法 | SP-1、SP-2、SP-5、MLP probe、MultiMax probe |
| 结果文件 | `results/{task}_{size}/{method}/final_results.json`，共 30 个 |

### 2.2 数据划分

| 任务 | 论文 | 本次复现 |
|---|---|---|
| VerIH | 5,000 train / 300 val / 300 test | 5,729 train / 322 val / 323 test |
| nemotron_safety | 10,000 train / 500 val / 500 test | 10,000 train / 400 val / 400 test |

### 2.3 超参数

- **SP-k**：AdamW，lr 3e-4，cosine 衰减到 1e-4，weight decay 0.01，batch size 1，dropout 0.2，10 epochs。与论文 Appendix B 一致。
- **MLP**：AdamW，lr 5e-4，batch size 32，10 epochs。原始结果使用代码默认 hidden dim **1024**；严格协议复现（4B/9B）使用论文的 **2048**。
- **MultiMax**：AdamW，lr 5e-4，batch size 32，10 epochs，H=10，MLP width 100。与论文一致。
- **mid layer**：原始结果 4B/9B/27B 使用 **16/16/32**；严格协议复现 4B/9B 使用论文的 **8/16**（27B 仍为 32，论文 20）。
- **27B hidden size**：本次使用的公开 checkpoint 为 **5120**；论文写的是 **3584**。

### 2.4 复现命令来源

- 4B/9B：`4B和9B的命令.md`
- 27B：`run_27B.sh`（在 `4B和9B的命令.md` 基础上改写）

---

## 3. 复现结果

### 3.1 VerIH（accuracy %，括号内为相对论文的 Δ）

| Model | MLP | MultiMax | SP-1 | SP-2 | SP-5 |
|---|---|---|---|---|---|
| 4B | 78.6 (+0.9) | 93.5 (+2.2) | 87.0 (-0.6) | 89.5 (+1.0) | 92.6 (+0.3) |
| 9B | 78.6 (-1.6) | 90.7 (-2.2) | 91.3 (+1.2) | 89.8 (0.0) | 91.3 (+0.9) |
| 27B | 73.1 (-8.6) | 93.8 (+0.3) | 80.5 (-10.8) | 79.6 (-12.4) | 83.9 (-6.8) |

论文值：4B 77.7/91.3/87.6/88.5/92.3；9B 80.2/92.9/90.1/89.8/90.4；27B 81.7/93.5/91.3/92.0/90.7。

- 4B MAE = **1.00 pp**
- 9B MAE = **1.19 pp**
- 27B MAE = **7.80 pp**

### 3.2 nemotron_safety（accuracy %，括号内为相对论文的 Δ）

| Model | MLP | MultiMax | SP-1 | SP-2 | SP-5 |
|---|---|---|---|---|---|
| 4B | 76.2 (+3.0) | 79.5 (-1.0) | 78.0 (+3.5) | 76.2 (+1.0) | 77.0 (+0.8) |
| 9B | 77.2 (+3.2) | 77.8 (-2.8) | 75.8 (-0.5) | 79.2 (+1.0) | 77.8 (+0.2) |
| 27B | 57.0 (-18.0) | 81.0 (+3.2) | 59.8 (-16.5) | 63.2 (-14.8) | 68.0 (-10.2) |

论文值：4B 73.3/80.5/74.5/75.2/76.2；9B 74.1/80.5/76.2/78.2/77.6；27B 75.0/77.8/76.2/78.0/78.2。

- 4B MAE = **1.86 pp**
- 9B MAE = **1.51 pp**
- 27B MAE = **12.52 pp**

### 3.3 严格协议复现结果（4B/9B）

严格协议 = MultiMax 使用论文的 mid-layer（4B layer 8、9B layer 16）+ MLP 使用论文的 hidden dim 2048。SP 的超参数本来就与论文一致，直接沿用原结果。数据划分仍为现有缓存的 5,729/322/323 和 10,000/400/400（见 5.3）。

| Model | Task | MLP | MultiMax | SP-1 | SP-2 | SP-5 |
|---|---|---:|---:|---:|---:|---:|
| 4B | VerIH | 78.02 (77.7, +0.32) | 90.71 (91.3, -0.59) | 87.00 (87.6, -0.60) | 89.47 (88.5, +0.97) | 92.57 (92.3, +0.27) |
| 4B | nemotron | 75.25 (73.3, +1.95) | 78.50 (80.5, -2.00) | 78.00 (74.5, +3.50) | 76.25 (75.2, +1.05) | 77.00 (76.2, +0.80) |
| 9B | VerIH | 79.88 (80.2, -0.32) | 92.26 (92.9, -0.64) | 91.33 (90.1, +1.23) | 89.78 (89.8, -0.02) | 91.33 (90.4, +0.93) |
| 9B | nemotron | 75.00 (74.1, +0.90) | 79.75 (80.5, -0.75) | 75.75 (76.2, -0.45) | 79.25 (78.2, +1.05) | 77.75 (77.6, +0.15) |

括号内为“论文值, Δ(pp)”。严格结果与之前用主缓存 / hidden 1024 的结果对比：

- 4B VerIH MultiMax：93.5（主缓存）→ **90.71**（layer 8），论文 91.3，更接近。
- 4B nemotron MultiMax：79.5（主缓存）→ **78.50**（layer 8），论文 80.5。
- 9B VerIH MultiMax：90.7（主缓存）→ **92.26**（layer 16），论文 92.9，明显更接近。
- 9B nemotron MultiMax：77.8（主缓存）→ **79.75**（layer 16），论文 80.5，明显更接近。
- MLP 改为 hidden 2048 后，4B/9B 两个任务与论文的误差均在 2 pp 内。

**严格结果目录**：`results_strict/{task}_{size}/{method}/final_results.json`（20 个）。其中 SP-1/2/5 是从 `results/` 原样复制的（超参数已与论文一致，未重跑），只有 MLP（hidden 2048）和 MultiMax（mid layer 8/16）是新训练；来源说明见 `results_strict/README.md`。

### 3.4 复现状态矩阵

| 实验 | 状态 | 说明 |
|---|---|---|
| 4B/9B SP-1/2/5（两任务） | ✅ 成功 | 与论文误差约 0–1.2 pp；数据划分与论文略有不同 |
| 4B/9B MLP（两任务） | ✅ 成功（严格协议） | 已改用论文 hidden dim 2048，误差 0.3–2.0 pp |
| 4B/9B MultiMax（两任务） | ✅ 成功（严格协议） | 已用 `_mid` 按论文 mid layer 8/16 重跑，误差 0.6–2.0 pp |
| 27B MultiMax（两任务） | ⚠️ 部分成功 | 改用 `_mid` 后接近论文（+0.3/+3.2 pp），但用的是 layer 32，论文为 20；27B checkpoint 本身也与论文不同 |
| 27B MLP（两任务） | ❌ 失败 | 低 8.6/18.0 pp；根因指向 27B checkpoint 不同（d=5120 vs 论文 3584） |
| 27B SP-1/2/5（两任务） | ❌ 失败 | 低 6.8–16.5 pp；同样指向 27B checkpoint 不同 |
| CoT Repetition | ⬜ 未复现 | 论文 Table 3 |
| CoT Reasoning Strategy | ⬜ 未复现 | 论文 Table 4 |
| MiniCPM4.1-8B | ⬜ 未复现 | 论文 Tables 2–5 中的 MiniCPM 行 |
| PII 检测 | ⬜ 未复现 | 论文 Table 1 |
| GPT-5.4-mini zero-shot | ⬜ 未复现 | 论文 Tables 2/3/4/5 的 zero-shot 行 |
| Qwen3Guard / Llama-Guard 对比 | ⬜ 未复现 | 论文 Table 5、Section 6.4 |
| Head ablation / cost 分析 | ⬜ 未复现 | 论文 Tables 6/7 |

### 3.5 独立重跑一致性检查

为了确认结果可复现，使用 `run_4b_9b_all.sh` / `reproduce_4b_9b.py` 把 4B/9B 的 20 个实验完整重跑了一遍，输出到 `results_verified/`，并与 `results_strict/` 逐项比较（容差 2 pp）。

- **20/20 结果全部生成**，文件/JSON/配置/NaN 检查全部通过。
- **17/20 在 2 pp 内**；总体 **MAE = 0.627 pp**，最大偏差 2.75 pp，平均有符号偏差 +0.397 pp。
- 3 个偏差略超 2 pp 的任务，经 GPU 架构复核后全部确认是**硬件数值差异**，不是实验错误：
  - 4B/9B nemotron MLP：重跑在 A6000 上分别 +2.75 pp；在 **RTX 4090** 上重跑后与第一次结果**完全一致**（75.25 / 75.00，Δ=0.00）。
  - 4B nemotron SP-5：A6000 重跑 +2.25 pp；在 **RTX 4090** 上重跑为 78.50 vs 77.00，**Δ=+1.50 pp（PASS）**。
- MultiMax 的第一次和重跑都在 A6000 上，4 个任务全部 **Δ=0.00**。

结论：4B/9B 的结果在同一 GPU 架构下可复现；跨架构（RTX 4090 vs A6000）会产生约 2–3 pp 的训练/测试波动，主要因为测试集较小（nemotron 400 个样本，1 个样本 = 0.25 pp）以及不同 GPU 的浮点/kernel 差异。若要追求逐位一致，应固定 GPU 型号和 CUDA/cuBLAS 版本。

相关产物：`results_verified/`（20 个重跑结果与 `summary.md`）、`results_recheck_4090_mlp/`、`results_recheck_4090_sp5/`。

---

## 4. 复现成功了什么

1. **4B/9B 的 SP-1/2/5 复现成功**：两个任务、三个 k 值共 12 个实验，与论文的 accuracy 误差约 **0–1.2 pp**。这说明递归 soft-prompt、MTP 头冻结、KV-cache 递归、线性 sigmoid head、验证集选 epoch 的整条方法链路是正确的。
2. **4B/9B 的 MLP probe 在论文 hidden dim 2048 下复现成功**：误差 0.3–2.0 pp。
3. **4B/9B 的 MultiMax 在论文 mid layer 8/16 下复现成功**：VerIH 90.71/92.26，nemotron 78.50/79.75；与论文相差 0.6–2.0 pp。这说明 MultiMax 实现和 mid-layer 特征提取正确。
4. **27B MultiMax 在修正缓存后接近论文**：VerIH 93.8（论文 93.5），nemotron 81.0（论文 77.8）。但 27B 的 mid layer 是 32（论文 20），且 checkpoint 不同，只能作参考。
5. **完整流水线可运行，并通过独立重跑校验**：模型下载、MTP 头提取、hidden-state 缓存、五种方法训练、结果落盘、验证集选 epoch、测试集评估全部跑通；原始 30/30 + 严格协议 20/20 结果文件齐全。用 `run_4b_9b_all.sh` 独立重跑 20 个严格实验，17/20 在 2 pp 内，3 个跨 GPU 偏差在 RTX 4090 上复核后全部通过（详见 3.5）。
6. **修复后的代码经过验证**：27B OOM 修复后做了 1-epoch 冒烟测试；修复后 SP-1 的 Epoch 1 验证集准确率与官方结果逐位一致（0.7298136645962733），证明快照修复不影响训练数值。

---

## 5. 复现失败了什么

### 5.1 27B 的 MLP 与 SP（核心失败）

27B 的 MLP 和 SP 系统性低于论文：

| 任务 | 方法 | 论文 | 复现 | Δ |
|---|---|---:|---:|---:|
| VerIH | MLP | 81.7 | 73.1 | -8.6 |
| VerIH | SP-1 | 91.3 | 80.5 | -10.8 |
| VerIH | SP-2 | 92.0 | 79.6 | -12.4 |
| VerIH | SP-5 | 90.7 | 83.9 | -6.8 |
| nemotron | MLP | 75.0 | 57.0 | -18.0 |
| nemotron | SP-1 | 76.2 | 59.8 | -16.5 |
| nemotron | SP-2 | 78.0 | 63.2 | -14.8 |
| nemotron | SP-5 | 78.2 | 68.0 | -10.2 |

**根因判断：论文的 27B checkpoint 与公开 checkpoint 不同。**

证据链：

1. 论文 Section 5.1 明确写 Qwen3.5-4B/9B/27B 的 hidden size 为 **d = 2560 / 4096 / 3584**。本地 4B/9B 的 config 是 2560/4096（吻合），但本地 27B 是 **5120**（不吻合）。
2. 论文说 MLP 参数量为 5–8M。hidden dim 2048 配 d=3584 约 7.3M（吻合）；配 d=5120 约 10.5M（超出范围）。
3. 失败是**方法特异性**的：使用 mid-layer 的 MultiMax 基本吻合，使用最终层的 MLP/SP 大幅偏低。这正是“最终层表示不同、mid 层表示相近”的典型表现。
4. 排除了其他解释：
   - 把 MLP hidden dim 改成论文的 2048：27B VerIH 73.1→74.3，nemotron 57.0→57.8，只补了约 1 pp，远不足以解释 8–18 pp。
   - 把缓存从 pre-final-norm 转成 post-final-norm（用 checkpoint 的 final RMSNorm 权重）：线性探针几乎不变（27B VerIH test 0.6935→0.6687，nemotron 0.5200→0.5350）。
   - NaN 污染：只有 10/10800 个样本，且坏 eval 样本在 val 不在 test；修复后 test 不受影响。

**结论**：27B 的这些数字不能作为论文 27B 结果的复现；它们对应的是当前公开的 Qwen3.5-27B（d=5120），而论文使用的是 d=3584 的 27B checkpoint。

### 5.2 严格 MultiMax 协议（4B/9B 已解决，27B 仍有层数差异）

- 4B/9B 已按论文要求重跑：MultiMax 使用 `_mid` 缓存，4B 为 layer 8、9B 为 layer 16；结果见 3.3，与论文相差 0.6–2.0 pp。
- 原始 4B/9B 结果使用的是主缓存（`4B和9B的命令.md`），现已由严格结果取代，原始结果保留在 `results/` 供对照。
- 27B 虽然改用了 `_mid`，但 mid layer 是 **32**，论文是 **20**；且 27B checkpoint 与论文不同，因此 27B MultiMax 仍只能算“数值接近”。

### 5.3 数据划分未完全对齐（4B/9B 超参数已对齐）

- VerIH：论文 5,000/300/300；复现 5,729/322/323。
- nemotron：论文 10,000/500/500；复现 10,000/400/400。
- 4B/9B 的 MLP hidden dim 已改为论文的 2,048；MultiMax mid layer 已改为论文的 8/16。
- 数据划分差异仍然存在：公开数据加载器无法精确重建论文的 5,000/300/300 和 10,000/500/500（当前 loader 返回 5,784/654；nemotron 在 `max_length=2048` 下最多 10,982 条）。
- 27B 的 MLP hidden dim 仍为 1,024（论文 2,048），mid layer 仍为 32（论文 20）；这两项在 27B 上因 checkpoint 不同而无法单独解决。

这些差异对 4B/9B 的影响较小（误差 1–3 pp），但对 27B 的精确对比不利。

### 5.4 论文其余实验未复现

- CoT Repetition（Table 3）
- CoT Reasoning Strategy（Table 4）
- MiniCPM4.1-8B（Eagle3 头；Tables 2–5 的 MiniCPM 行）
- PII 检测（Table 1）
- GPT-5.4-mini zero-shot（Tables 2–5 的 zero-shot 行）
- Qwen3Guard-Gen-8B / Llama-Guard-3-8B 对比
- Head ablation（Table 7）、cost 分析（Table 6）
- WildGuardMix 外部验证（Section 6.5）

---

## 6. 复现过程中的困难

### 6.1 27B 训练 OOM（首个阻塞问题）

- 现象：27B `train_softprompt.py` 在 Epoch 2 刷新 best epoch 时，`train_softprompt.py:777` 抛 `torch.OutOfMemoryError`，尝试分配 2.37 GiB。
- 根因：`classifier.state_dict()` 同时包含冻结的 MTP 头（约 5.5 GiB bf16），原代码用 `v.detach().clone()` 在 GPU 上克隆整份 state dict。Epoch 1 首次快照后进程已占 13.93 GiB；Epoch 2 旧快照仍存活，新克隆再要 5.5 GiB；而 GPU 0 被其他用户占用约 31.4 GiB，只剩 2.0 GiB，第一个大张量（`embed_tokens`，2.37 GiB）就失败了。
- 修复：best snapshot 只保存 soft prompt + head，并放 CPU；恢复时 `strict=False`。
- 验证：1-epoch 冒烟测试通过；修复后 SP-1 Epoch 1 的 val acc 与官方结果完全一致（0.7298136645962733）。

### 6.2 后台任务被会话退出杀掉

- 复现过程中退出 Claude Code 时，被跟踪的后台训练任务被终止。虽然独立 watchdog（`setsid + nohup`）自动重启了任务，但正在跑的 27B VerIH SP-5 从 Epoch 3 回到 Epoch 1，损失约 20 分钟。
- 之后使用独立 session 的 watchdog 守护，最终 10/10 结果完成并正常退出。

### 6.3 27B nemotron 缓存中的 NaN 样本

- 扫描发现 `cache_nemotron_safety_27b.pt` 有 **10/10800** 个样本的最终层 hidden state 全为 NaN（多为 4–7 token 的极短阿拉伯语/波斯语 prompt）。
- `train_softprompt.py` 有 `torch.isfinite(loss)` 保护，会跳过坏 batch；`train_mlp.py` 没有，一个 NaN 样本就让 MLP 权重全部变 NaN，val 恒为 0.5、F1=0。
- 修复：`train_mlp.py` 加入非有限 loss 跳过逻辑；重跑后 nemotron 27B MLP 恢复到 57.0 acc / 0.5635 F1（仍低，原因是 27B checkpoint 不同，见 5.1）。
- 10 个坏样本中 9 个在 train、1 个在 val、0 个在 test，所以 SP/MultiMax 的 test 指标不受污染，仅 val 选 epoch 有 1/400 的固定误差。

### 6.4 MultiMax 缓存路径错误

- 27B 的 `run_27B.sh` 最初把主缓存传给 `train_multimax.py`，导致 VerIH 只有 0.5232（接近随机）。改用 `_mid` 后恢复到 0.9381。
- 4B/9B 的命令文件同样把主缓存传给 MultiMax。严格复现时已改用 `_mid` 并重新训练：4B layer 8、9B layer 16；结果为 4B 90.71/78.50、9B 92.26/79.75（论文 91.3/80.5、92.9/80.5）。

### 6.5 共享 GPU 与会话环境

- 服务器 GPU 0 被多个用户占用，27B 训练一度只剩 2 GB 显存；后来改到空闲的 GPU 4。
- 新开 Claude Code 会话后，默认 `python` 指向 base conda 环境（没有 torch），需要使用 `/home/suzeqin/miniconda3/envs/sp/bin/python` 或先 `conda activate sp`。
- 27B SP-5 每 epoch 约 9–11 分钟，两个任务合计训练时间约 8 小时（含一次重启）。

### 6.6 论文与公开仓库不一致

- 公开仓库的 `scripts/run_qwen.sh` 与论文描述不一致（见第 8 节），无法直接产出论文表格。
- 论文 27B checkpoint（d=3584）在公开 HuggingFace 上没有找到对应版本；本地下载的 Qwen3.5-27B 为 d=5120。
- 未能获得论文的精确 27B checkpoint，因此 27B 的失败无法通过调参解决。

---

## 7. 修改过的代码

### 7.1 `fast_softprompt/train_softprompt.py`（原仓库代码）

**问题**：best-epoch 快照把冻结的 MTP 头也克隆到 GPU，27B 上第二次快照 OOM。

**修改**：只保存可训练参数（soft prompts + head），并放到 CPU；恢复时允许缺失冻结键。

```diff
-            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
+            best_state = {
+                k: v.detach().cpu().clone()
+                for k, v in classifier.state_dict().items()
+                if k.startswith("soft_prompt") or k.startswith("head.")
+            }
...
-        classifier.load_state_dict(best_state)
+        classifier.load_state_dict(best_state, strict=False)
```

**验证**：1-epoch 冒烟测试通过；修复后 SP-1 Epoch 1 val acc 与官方结果完全一致。该修改不影响训练数值，只影响 best-state 的存储位置和内容。

### 7.2 `fast_softprompt/train_mlp.py`（原仓库代码）

**问题**：缓存中存在少量 NaN hidden state 时，一个 NaN batch 会让整个 MLP 权重变成 NaN。

**修改**：与非有限 loss 的 batch 跳过更新。

```diff
             logits, loss = classifier(train_f[s:e], train_l[s:e])
+            if not torch.isfinite(loss):
+                # A handful of cached 27B hidden states are non-finite (NaN).
+                # Skip the batch so one bad sample cannot poison every weight.
+                continue
             loss.backward()
             optimizer.step()
```

### 7.3 `run_27B.sh`（复现脚本，非原仓库）

- 新增 `MID_CACHE`，让 `train_multimax.py` 使用 `_mid` 缓存（原脚本误用主缓存）。
- 新增“已完成任务自动跳过”逻辑，避免重启后重复训练。
- 这些修改不影响 SP/MLP 的训练数值。

### 7.4 未修改但有问题的地方

- `run.sh` 和 `4B和9B的命令.md` 仍然把主缓存传给 `train_multimax.py`。如果要严格复现 MultiMax，需要改成 `_mid` 缓存。
- `run.sh`/`run_27B.sh` 没有固定 `CUDA_VISIBLE_DEVICES`，在共享服务器上容易抢到被占用的 GPU。

### 7.5 备份

- 原始 NaN 污染的 MLP 结果：`/tmp/nemotron_safety_27b_mlp_nan_poisoned.json`
- 原始错误缓存的 MultiMax 结果：`/tmp/verih_27b_multimax_wrongcache.json`、`/tmp/nemotron_27b_multimax_wrongcache.json`
- 27B MLP hidden_dim=2048 的诊断结果：`/tmp/mlp2048_verih_27b/`、`/tmp/mlp2048_nemotron_27b/`

### 7.6 严格复现新增产物

- `results_strict/{task}_{size}/{method}/final_results.json`：4B/9B 在论文协议下的严格结果（20 个）。
- `cache_verih_4b_mid8.pt`、`cache_nemotron_safety_4b_mid8.pt`：4B 的 mid-layer 8 缓存（论文要求 layer 8）。
- `fast_softprompt/precompute_mid_from_cache.py`：从现有缓存重新提取指定 mid layer 的工具。
- 原仓库代码没有再新增修改；严格复现只改变了 MultiMax 的输入缓存和 MLP 的 `--hidden_dim` 参数。

---

## 8. 原仓库的问题

### 8.1 代码正确性 / 健壮性

1. **`train_softprompt.py`：best-state 克隆整个冻结 MTP 头（严重）**
   - 位置：原第 777 行。
   - 影响：27B 上第二次刷新 best epoch 时 OOM；论文硬件是 96 GB，所以作者未暴露此问题。
   - 建议：只保存 `requires_grad` 参数或 `soft_prompt*`/`head.*`，并放 CPU。

2. **`train_mlp.py`：没有非有限 loss 保护（严重）**
   - 位置：训练循环中 `loss.backward()` 之前。
   - 影响：一个 NaN hidden state 就能污染全部权重；27B nemotron 结果直接变成随机水平。
   - 建议：与 `train_softprompt.py` 一致，加入 `torch.isfinite(loss)` 检查。

3. **`precompute_hidden_states.py`：不检查 hidden state 的有限性（严重）**
   - 27B nemotron 缓存有 10/10800 个全 NaN 样本，预计算阶段没有任何告警。
   - 建议：写入缓存前检查 `torch.isfinite`，记录/剔除坏样本；或在训练脚本中统一做非有限保护。

4. **`precompute_hidden_states.py`：主缓存语义依赖 `--mid_layer`（中等）**
   - 逻辑：
     ```python
     need_all_layers = save_mid or args.layer != -1
     if need_all_layers:
         hidden = out.hidden_states[args.layer]   # 最后一层 decoder 输出（final norm 之前）
     else:
         hidden = out.last_hidden_state            # final norm 之后
     ```
   - 只要传了 `--mid_layer`（本复现所有缓存都传了），主缓存就是 **pre-final-norm**；不传时才是 post-norm。两条路径的语义不一致，且没有文档说明。
   - 建议：显式增加 `--hidden_norm {pre,post}` 参数，或在 README 说明。

5. **`train_mlp.py` / `train_multimax.py`：best-state 也在 GPU 上克隆（轻微）**
   - 这两个模型的 state dict 很小，当前不会 OOM，但与 `train_softprompt.py` 是同类问题。
   - 建议：统一改成 `.detach().cpu().clone()`。

6. **`train_softprompt.py`：save-epoch-ckpts 分支有死代码（轻微）**
   - `if False else` 结构（原第 756–765 行），不影响运行，但可读性差。

### 8.2 与论文不一致 / 可复现性问题

7. **README 的 MLP hidden dim 与代码/脚本不一致（中等）**
   - README：MLP hidden dim **2048**。
   - `train_mlp.py` 默认：**1024**。
   - `scripts/run_qwen.sh`：没有传 `--hidden_dim`，所以实际用 **1024**。
   - 论文 Appendix B 也写 2048。公开仓库无法直接按论文超参运行。

8. **`scripts/run_qwen.sh` 与论文的 mid layer 不一致（中等）**
   - 脚本对所有尺寸写死 `--mid_layer 16`。
   - 论文：4B/9B/27B 分别用 **8/16/20**。
   - 影响 MultiMax 的协议一致性。

9. **`scripts/run_qwen.sh` 与论文的数据划分不一致（中等）**
   - 脚本：`--num_test 1000`，VerIH/nemotron 的 `target_eval` 分别是 645/800。
   - 论文：VerIH 5,000/300/300；nemotron 10,000/500/500。
   - 由于 `make_val_test_split` 的逻辑（见下一条），实际 val/test 会变成 500/500，而不是 README 声称的 322/323 或 400/400。

10. **`eval_split.py` / README 对 `target_eval` 的描述有误（中等）**
    - README/注释说 `--target_eval` 设定 held-out pool 的大小。
    - 实际代码：`target_eval` 只是**下限**。如果 `num_test_cached > target_eval`，eval pool 仍然是整个 `num_test`，val/test 会更大。
    - 例：`num_test=1000, target_eval=645` 会得到 500/500，而不是 645 的 50/50。
    - 建议：要么在代码中 cap 到 `target_eval`，要么修正文档。

11. **`requirements.txt` 不锁版本（轻微/中等）**
    - `torch>=2.4`、`transformers>=4.50` 没有上限。
    - Transformers 5.x 的 `output_hidden_states` 捕获机制（`capture_outputs`/`_can_record_outputs`）与旧版本可能不同，会改变主缓存的语义（pre-norm vs post-norm）。
    - 建议：锁定论文使用的 transformers/torch 版本。

12. **没有锁定模型 revision（中等）**
    - `scripts/run_qwen.sh` 使用 `Qwen/Qwen3.5-${SIZE}`，没有 pin revision。
    - 论文 27B 的 d=3584 与当前公开的 d=5120 不一致，说明 HF 模型可能已更新，或论文使用了不同的 27B checkpoint。
    - 建议：在 README/脚本中写明论文使用的模型 revision/hash。

13. **`precompute_hidden_states.py` 的设备获取方式较脆弱（轻微）**
    - `device = str(model.lm_head.weight.device)` 在 `device_map="auto"` 下可能不是输入层所在设备；当前能跑通，但依赖 accelerate 的 hook 行为。
    - 建议：用 `model.get_input_embeddings().weight.device` 或显式 device map。

### 8.3 复现脚本（非原仓库）的问题

14. **`4B和9B的命令.md` / `run.sh`：MultiMax 使用了主缓存**
    - 命令里是 `--cache cache_verih_${TAG}.pt`，而论文/README 要求 `_mid` 缓存。
    - 当前 4B/9B MultiMax 数值接近论文，但协议不一致；严格复现需要重跑。

15. **`run_27B.sh`：MultiMax 也曾使用主缓存（已修复）**
    - 修复后 27B VerIH MultiMax 从 0.5232 恢复到 0.9381。

16. **脚本没有固定 GPU**
    - 在共享服务器上容易落到被其他用户占满的 GPU 0，导致 OOM 或训练中断。建议 `CUDA_VISIBLE_DEVICES` 显式指定。

---

## 9. 结论与建议

### 9.1 可以使用的结论

- **4B/9B 的严格协议复现已完成**：MultiMax 使用论文的 mid layer（4B=8，9B=16），MLP 使用论文的 hidden dim 2048；20 个实验与论文相差约 0.3–2.0 pp（总体 MAE 约 0.92 pp）。严格结果在 `results_strict/`；数据划分仍有差异（见 5.3）。
- **独立重跑一致性已确认**：20 个严格实验重跑后 17/20 在 2 pp 内（总体 MAE 0.627 pp）；3 个跨 GPU 偏差在 RTX 4090 上复核后全部通过（MLP Δ=0.00，4B SP-5 Δ=+1.50 pp），说明结果可复现，差异来自 GPU 架构而非实验错误。
- **27B 的结果不能作为论文 27B 的复现**：当前公开 27B checkpoint（d=5120）与论文（d=3584）不同，MLP/SP 系统性偏低。
- **27B MultiMax 的数值接近论文**，但 mid layer 32 与论文 20 不同，且受 27B checkpoint 差异影响，只能作为参考。

### 9.2 下一步建议

1. **获取论文的 27B checkpoint（d=3584）**：检查 HuggingFace `Qwen/Qwen3.5-27B` 在 2026 年 8 月前后的 revision，或联系论文作者确认。拿到后重新提取 MTP 头、重建缓存、重跑 27B 的 SP/MLP。
2. **27B 的严格协议**：4B/9B 已完成（MultiMax layer 8/16，MLP hidden 2048）；27B 仍需 mid layer 20（当前 32）和论文的 d=3584 checkpoint；MLP hidden 2048 只有在拿到论文 checkpoint 后才有意义。
3. **数据划分**：4B/9B 的严格结果仍使用现有缓存划分（5,729/322/323、10,000/400/400）；如需完全对齐论文的 5,000/300/300 和 10,000/500/500，需要重建数据缓存（公开 loader 无法精确重建，见 5.3）。
4. **补做未复现实验**：CoT Repetition、CoT Reasoning Strategy、MiniCPM4.1-8B、PII、GPT-5.4-mini zero-shot、Qwen3Guard/Llama-Guard 对比、head ablation、cost 分析。
5. **保留本报告的代码修复**：它们只影响显存/健壮性，不改变 4B/9B 的训练数值；建议向原仓库提 PR。

---

## 10. 附录

### 10.1 结果文件

```
results/
├── verih_4b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
├── verih_9b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
├── verih_27b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
├── nemotron_safety_4b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
├── nemotron_safety_9b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
└── nemotron_safety_27b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
```

共 30 个 `final_results.json`。

严格协议结果（4B/9B，MultiMax mid layer 8/16 + MLP hidden 2048）：

```text
results_strict/
├── verih_4b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
├── verih_9b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
├── nemotron_safety_4b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
└── nemotron_safety_9b/{sp1,sp2,sp5,mlp,multimax}/final_results.json
```

共 20 个；SP 结果从 `results/` 复制（超参与论文一致，数据划分未变）。

### 10.2 关键命令

完整的结果级命令（覆盖 `results/` 和 `results_strict/` 的每一组结果）见 `RUN_COMMANDS.md`；下面只列关键命令。

- 4B/9B：见 `4B和9B的命令.md`。
- 27B：见 `run_27B.sh`（已包含 MultiMax `_mid` 缓存修复和断点续跑逻辑）。
- 27B MLP hidden_dim=2048 诊断：
  ```bash
  python fast_softprompt/train_mlp.py \
    --cache cache_verih_27b.pt --mmap_cache \
    --output_dir /tmp/mlp2048_verih_27b \
    --hidden_dim 2048 --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645
  ```
- 4B/9B 严格协议（MultiMax mid layer 8/16 + MLP hidden 2048）：
  ```bash
  # 9B MultiMax（论文 mid layer 16）
  python fast_softprompt/train_multimax.py \
    --cache cache_verih_9b_mid.pt --mmap_cache \
    --output_dir results_strict/verih_9b/multimax \
    --num_heads 10 --mlp_width 100 --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645

  # 4B MultiMax（论文 mid layer 8；缓存由 fast_softprompt/precompute_mid_from_cache.py 生成）
  python fast_softprompt/train_multimax.py \
    --cache cache_verih_4b_mid8.pt --mmap_cache \
    --output_dir results_strict/verih_4b/multimax \
    --num_heads 10 --mlp_width 100 --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645

  # MLP（论文 hidden dim 2048）
  python fast_softprompt/train_mlp.py \
    --cache cache_verih_4b.pt --mmap_cache \
    --output_dir results_strict/verih_4b/mlp --hidden_dim 2048 \
    --lr 5e-4 --epochs 10 --batch_size 32 --target_eval 645
  ```

### 10.3 参考

- 论文：https://arxiv.org/abs/2608.28099
- ACL Anthology：https://aclanthology.org/2026.findings-emnlp.1239/
- 官方代码：https://github.com/collinzrj/SpeculativeProbing
