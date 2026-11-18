# WebInstruct-Verified CoT 推理轨迹复现：方案、进度与下一步

> 更新时间：2026-11-18 14:55 ｜ 分支：`CoT推理轨迹复现` ｜ 工作目录：`/data1/suzeqin/SpeculativeProbing`（脚本集中在 `cot_local/`）
>
> 本文档是本任务的**交接文档**，供 token 恢复后的新会话直接续跑使用。
> **本次续跑发现了比 OOM 更严重的问题（标注解析 bug + 教师模型语料缺陷），详见 §9。当前正在按干净标签重跑 ④⑤⑥。**

---

## 1. 任务定义

复现《Speculative Probing》（论文 PDF 在仓库根目录 `Speculative Probing.pdf`）中 **WebInstruct-verified** benchmark 上、需要先生成 CoT 推理轨迹的两个实验（论文 §6.2 CoT Repetition = Table 3、§6.3 CoT Reasoning Strategy = Table 4）：

| 维度 | 内容 |
|---|---|
| 模型（3 个，学生模型） | Qwen3.5-4B、Qwen3.5-9B、MiniCPM4.1-8B |
| 方法（5 种） | SP1、SP2、SP5、MLP、MM |
| Benchmark（1 个） | WebInstruct-verified（CoT 轨迹需本地生成） |
| **产出** | **3 × 5 × 1 = 15 组测试数据 × 2 张表 = 30 个数字**，与论文逐项对比（compile_results.py 已内置论文数值） |
| 教师模型（CoT 生成+标注裁判） | 论文：闭源 API Qwen3.5-Plus 生成 + GPT-5.4 标注；本地：**Qwen3.5-122B-A10B-GPTQ-Int4** 一站到底（Qwen3.5 系列开源旗舰 MoE，GPTQ Int4，效果最接近 Qwen3.5-Plus 的可本地部署开源模型） |
| 权重目录 | `/data1/suzeqin/SpeculativeProbing/models`（已全部下载） |

方法含义（以论文/README 为准，注意 **SP-k 不是"在第 k 步探测"**）：
- **SP-k（SP1/SP2/SP5）**：Speculative Probing。冻结学生模型的 MTP 头（Qwen3.5，预训练自带）或 Eagle3 头（MiniCPM4.1），在序列尾部追加 **k 个可学习递归软提示向量**，从 MTP/Eagle3 头池化输出上接线性二分类头。每任务仅训 ~16-20K 参数。Qwen 用 `train_softprompt.py`，MiniCPM 用 `train_eagle3_minicpm.py`；
- **MLP**：末 token 末层隐状态的 2 层 MLP 基线（`train_mlp.py`，读 `cache_<task>_<model>.pt`）；
- **MM（MultiMax）**：Kramár et al. 的位置感知探针（`train_multimax.py`，读中间层缓存 `cache_<task>_<model>_mid.pt`）。

两个任务的标签（同一份轨迹语料，两套标注）：
- **cot_repetitive**：`<think>` 里是否出现"无意义重复思考"（yes/no 二分类，论文 Appendix A.2 裁判提示词）；
- **cot_branched**：推理结构是 linear（单链）还是 branched（多分支/回溯）。

### 论文参考数值（compile_results.py 已内置，格式：复现值/论文值）

Table 3 CoT Repetition（准确率 %）：

| 模型 | MLP | MM | SP-1 | SP-2 | SP-5 |
|---|---|---|---|---|---|
| Qwen3.5-4B | 87.5 | 87.0 | 88.0 | 87.5 | 89.0 |
| Qwen3.5-9B | 89.5 | 87.5 | 91.5 | 89.5 | 89.5 |
| MiniCPM4.1-8B | 83.5 | 82.5 | 86.0 | 85.0 | 87.0 |

Table 4 CoT Reasoning Strategy（准确率 %）：

| 模型 | MLP | MM | SP-1 | SP-2 | SP-5 |
|---|---|---|---|---|---|
| Qwen3.5-4B | 73.0 | 85.5 | 83.0 | 81.5 | 82.5 |
| Qwen3.5-9B | 76.5 | 87.0 | 84.5 | 83.0 | 83.5 |
| MiniCPM4.1-8B | 68.0 | 78.5 | 66.5 | 65.5 | 64.5 |

（论文协议：~10k 条 Qwen3.5-Plus 轨迹 → 过滤 >8192 token → 平衡正负 → ~4950 训练 + 200/200 验证/测试；27B 行不做，零样本 GPT-5.4-mini 基线不做——无 API。）

---

## 2. 环境与资源快照（2026-11-17 11:54 实测）

### GPU（8×48GB）

| GPU | 状态 | 占用 |
|---|---|---|
| 0-1（RTX 4090） | **他人进程**（`...d-jev/.venv-qwen38-pilot`），勿动 | 25-28GB |
| 2-3（4090） | 本任务 vLLM **实例 B**（TP=2，标注用，端口 8001） | ~44GB×2 |
| 4-7（A6000） | 本任务 vLLM **实例 A**（TP=4，轨迹生成用，端口 8000） | ~48GB×4 |

phase3 启动时会自动 `pkill` 两个 vLLM 实例、释放 2-7 号卡给 precompute/训练。

### 磁盘

`/data1` 7TB 已用 ~2.2TB，剩 **~4.5TB**。预计本次新增缓存 ~2.2-2.8TB（4B 两任务 ~96GB、9B ~150GB、mcpm ~1.3TB，见 §6），够用但紧张；**跑完一个模型可先删另一任务的中间文件再核对**。

### Python 环境（两套，别混用）

| env | 用途 | 关键包 |
|---|---|---|
| `~/miniconda3/envs/sp`（python 3.10） | 生成/标注客户端、precompute、全部训练 | torch 2.8.0+cu126、transformers 5.20.2、flash-linear-attention、httpx、**causal-conv1d 1.7.0**（从 GitHub release 装 cp310+cu12torch2.8 预编译 wheel，源码编译会失败：无 nvcc） |
| `~/miniconda3/envs/serve27`（python 3.12） | vLLM 推理服务 | **vllm 0.27.1+cu129**（GitHub release wheel）+ torch 2.13.0+cu129 + torchvision/torchaudio 0.28/2.11+cu129 |

> vLLM 版本陷阱（已解决，勿重装）：pip 上的 vllm 0.27+/0.33 自带 torch **cu130**，驱动 560.35（CUDA 12.6）初始化即报 "driver too old"。解决 = 从 pytorch 官方 cu129 索引装 torch 全家桶（cu129 走 CUDA 12 次版本兼容），再从 GitHub releases 装 `vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`，并 `pip uninstall torchcodec`（其 .so 依赖 cu13 无法加载）。qwen3_5 架构在 vllm ≥0.27.1 才支持。
> 启动 vLLM 必须先 `export PATH=~/miniconda3/envs/serve27/bin:$PATH`（子进程要找 ninja，否则 FileNotFound）。

### 模型（全部在 `models/`，无需再下载）

| 目录 | 用途 |
|---|---|
| `Qwen3.5-122B-A10B-GPTQ-Int4`（74GB，39 分片） | 教师模型：轨迹生成 + 两任务标注裁判 |
| `Qwen3.5-4B` / `Qwen3.5-9B` | 学生模型（qwen3_5 混合线性注意力架构，需 causal-conv1d） |
| `MiniCPM4.1-8B` / `MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16` | 学生模型 + Eagle3 头（E3DIR=`models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16`） |
| `Qwen3.5-4B-mtp` / `Qwen3.5-9B-mtp` | 已提取的 MTP 头 |

---

## 3. 技术方案（整体流水线）

```
① 题集        TIGER-Lab/WebInstruct-verified train，seed=42 抽 18000 题
      │
② CoT 生成    cot_local/gen_traces.py → 实例A(端口8000)
      │       复用仓库脚本逻辑；关键改动：max_completion_tokens=8192（原4096下72%轨迹
      │       think 中途截断被弃）、容忍 chat 模板把 <think> 开标签写进 prompt 的提取
      │       输出：cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4.jsonl
      │       watchdog.sh：语料 ≥13000 条时 kill 生成客户端并 touch cot_local/.gen_done
      │       （13000 条 ≈ 9.6k valid ≈ 论文 ~10k 轨迹语料；断点可续，error ~25% 是
      │        >8192 token 的截断丢弃，属协议内行为）
      │
③ 标注        cot_local/label_traces.py（裁判=同一 122B，开 thinking，只取 </think>
      │       之后的回答解析）× label_loop.sh 滚动追新（实例B，两任务各 12 并发）
      │       标注协议：论文原版 judge prompt（branched=linear/branched，其余=yes/no），
      │       temperature=0；prompt 超长(>+1024 token 到 9216)客户端直接跳过
      │       （这些轨迹过不了 precompute 的 8192 过滤，标注了也用不上）
      │       输出：…_labeled_repetitive_thinking.jsonl / …_labeled_branched_thinking.jsonl
      │       phase2.sh：等 .gen_done 后在实例A（100 并发）跑 final_labeling.sh 收尾
      │       （3 轮重试），两文件覆盖全语料后 touch cot_local/.labels_done
      │
④ 隐状态      phase3.sh：先做闸门校验（corpus≥12000、rep_pos≥1500、br_pos≥1500，
      │       按 11:49 的正率 25%/43% 预计 ~3300/~5400，稳过），kill 两个 vLLM，
      │       然后 6 个 (模型×任务) precompute 作业并行上 GPU 2-7：
      │       run_precompute_one.sh <gpu> <4b|9b|mcpm> <cot_repetitive|cot_branched>
      │       协议：--num_train 5000 --num_test 400 --max_length 8192 --mid_layer 16
      │       （与 scripts/run_qwen.sh 的 cot 分支完全一致），batch_size 8（纯吞吐优化）
      │       产出：cache_<task>_<model>.pt(+.hidden)、_mid.pt；mcpm 另有 _eagle3.pt
      │
⑤ 训练        phase3.sh 继续：6 个 run_train.sh <gpu> <model> <task> 并行
      │       SP1/2/5：EPOCHS=4、TE=400、Qwen 用 train_softprompt.py(lr 3e-4→1e-4,
      │       batch 1, dropout 0.2, recursive, linear head)，mcpm 用
      │       train_eagle3_minicpm.py(--base_model models/MiniCPM4.1-8B 必须！
      │       默认会去 HF cache 找 embed_tokens 而挂掉)
      │       MLP/MM：epochs 10、batch 32、lr 5e-4（MM 读 _mid 缓存）
      │       产出：results/<task>_<model>/{sp1,sp2,sp5,mlp,multimax}/final_results.json
      │       （准确率在 json 的 metrics.accuracy，最佳 epoch 按 val 选）
      │
⑥ 汇总        python cot_local/compile_results.py → cot_local/RESULTS.txt
              15 格/表 × 2 表，格式"复现值/论文值"，写报告
```

运维/监控脚本（均在 `cot_local/`）：
- `status.sh`：一次打印语料/两标注计数/正例数/两端吞吐；
- `logger.sh`：每 10 分钟往 `monitor.log` 追加一行（在跑）；
- `watchdog.sh`：**只负责到 13000 条停生成**（不自动拉起，挂了需手动，见 §5）；
- `phase2.sh` / `phase3.sh`：两个里程碑看护（标注收尾 → precompute+训练+汇总），已在后台等触发；
- `smoke/`：全链路冒烟（含 SP1/MLP/MM/E3SP1 小样本输出），已验证可通。

---

## 4. 当前进度（截至 11:54）

- ✅ 权重全部下载（教师 74GB + 3 学生 + Eagle3/MTP 头）；
- ✅ vLLM 环境趟平（0.27.1+cu129，驱动兼容），实例 A/B 双服务在跑；
- ✅ 全链路脚本冒烟通过（`smoke/` 有 SP1/MLP/MM/E3SP1 输出；标注裁判经合成正例 sanity 检验）；
- 🔄 **② 生成中**：客户端 `Generated 8300/17829 (valid: 6174, errors: 2126)`，语料落盘 **6285 条**；吞吐 ~1800 tok/s（4 卡极限），语料增速 ~855 条/h；
- 🔄 **③ 标注并行追新**：rep 3916 条（正例 996，**25.4%**）、br 3821 条（正例 1647，**43.1%**），与论文量级一致（branched 论文也偏多）；
- ⏳ ④⑤⑥ 由 phase3.sh 自动接力（等 .labels_done）。

### 预计时间线

| 里程碑 | 预计 |
|---|---|
| 语料到 13000、生成停止 | **今晚 ~19:45-20:00** |
| 实例 A 收尾标注（~9k 条 × 2 任务，100 并发） | +1.5~2h，~21:30-22:00 |
| 6 个 precompute 并行 | +1~2h（bs=8 实测 9B ~2900 tok/s；mcpm 写三份缓存更慢些） |
| 6 个训练单元并行（每单元 SP×3+MLP+MM） | +1~2h |
| **全部 30 个数字 + 对比表出炉** | **今晚 ~24:00 前后** |

---

## 5. 若中断怎么续（新会话操作手册）

1. **先看状态**：`bash cot_local/status.sh`、`tail -5 cot_local/monitor.log`、`tail -20 cot_local/phase2_main.log cot_local/phase3_main.log`。
2. **判活**：`pgrep -af "gen_traces|label_loop|phase2.sh|phase3.sh|watchdog"`；vLLM 存活看 `curl 127.0.0.1:8000/v1/models` 与 `:8001/v1/models`。
3. **常见恢复**：
   - 生成客户端挂了 → `nohup ~/miniconda3/envs/sp/bin/python cot_local/gen_traces.py --base-url http://127.0.0.1:8000/v1 --model Qwen3.5-122B-A10B-GPTQ-Int4 --num-samples 18000 --workers 128 >> cot_local/gen_traces.log 2>&1 &`（断点续跑，已生成题目自动跳过；注意 `export HF_ENDPOINT=https://hf-mirror.com`，否则 datasets 联网重试）；
   - vLLM 实例挂了 → 按 §2 环境说明重启（A：TP=4 于 GPU4-7，`--gpu-memory-utilization 0.97 --max-model-len 9216 --skip-mm-profiling`；B：TP=2 于 GPU2-3，另加 `--max-num-seqs 32`，否则 Mamba 缓存块不足起不来；都必须先 export PATH）；
   - 标注循环挂了 → `nohup bash cot_local/label_loop.sh <task> 8001 12 &`；
   - phase2/phase3 看护挂了 → `nohup bash cot_local/phase2.sh / phase3.sh > …_main.log 2>&1 &`（幂等：靠 .gen_done/.labels_done 标记与各脚本自身的 resume）。
4. **已完成 ②③ 想直接跑 ④⑤**：`touch cot_local/.gen_done cot_local/.labels_done` 后拉起 phase3.sh（注意先确认 gate 数值够，phase3 自带校验）。
5. **结果核对**：`cat cot_local/RESULTS.txt`；单格明细在各 `results/<task>_<model>/<method>/final_results.json` 的 `metrics.accuracy`。

## 6. 磁盘预算（phase3 产出，bf16 mmap，~5400 样本×~2500 tok 中位）

| 缓存 | 估算 |
|---|---|
| cache_cot_*_4b(.pt/.hidden) + _mid | 2 任务 ~96GB（verih 4B 7.7GB/份、cot 序列长得多） |
| cache_cot_*_9b + _mid | ~150GB |
| cache_cot_*_mcpm{,_mid,_eagle3} | ~1.2-1.4TB（eagle3 是 3 层拼接 12288 维，参考 verih 40GB/份） |
| **合计** | **~1.5-1.7TB**（剩 4.5TB，够；若紧可先跑 Qwen 两模型再跑 mcpm） |

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| 教师 122B-A10B-Int4 与 Qwen3.5-Plus 有差距 | 本地可部署的最强近似；报告注明教师差异为主要偏差来源。已 sanity 检验：裁判对合成正例判对、真实轨迹正率与论文同量级 |
| 生成 error ~25% | 均 >8192 token 截断丢弃（协议内）；13000 条目标已留余量（预计 valid ~9.6k） |
| 标注滞后 | phase2 在实例 A 上 100 并发收尾；实例 B 滚动追新防积压 |
| 磁盘不足 | §6 预算够；紧张时按模型分批跑 precompute |
| phase3 gate 失败中止 | 故意设计（防脏数据进训练）；按 phase3_main.log 提示人工核 corpus/正例数后重启 |
| MiniCPM 训练报 "no embed_tokens" | run_train.sh 已加 `--base_model models/MiniCPM4.1-8B`，勿删 |
| 结果与论文偏差大 | 先查：标签分布 vs 论文、test 集 200/200 平衡、mid_layer=16；trend 对齐优先于绝对值（教师模型不同） |

## 8. 关键路径速查

- 数据：`cot_inconsistency/data/generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4*.jsonl`（语料/两任务标注）
- 服务日志：`cot_local/vllm_122b.log`（实例A）、`vllm_122b_b.log`（实例B）｜ 生成：`gen_traces.log` ｜ 标注：`label_{rep,branched}*_p8001.log`
- 看护/监控：`watchdog.log`、`phase2_main.log`、`phase3_main.log`、`monitor.log`
- 训练实现：`fast_softprompt/train_{softprompt,eagle3_minicpm,mlp,multimax}.py`；缓存：`fast_softprompt/precompute_hidden_states{,_cot,_minicpm}.py`
- 结果：`results/cot_{repetitive,branched}_{4b,9b,mcpm}/`，汇总 `cot_local/RESULTS.txt`（编译脚本含论文对照值）
- 历史复现（verih/nemotron 等）的 cache/日志在仓库根目录，可作格式参照；**cot_local/ 尚未 git 提交，收尾时记得 add+commit**

---

## 9. 2026-11-18 续跑记录（重要：两个根因 + 干净标签重跑）

### 9.1 首次 phase3 的两个失败根因

**① OOM（表层问题，已修）**：6 路训练并行，每个进程 `torch.load(mmap=True)` 默认用
`MAP_PRIVATE` 映射 100-600GB 的 `.hidden`；该可写映射计入内核 commit limit
（`vm.overcommit_ratio=50` → CommitLimit 仅 ~282GiB），608GB 的
`cache_cot_repetitive_mcpm_eagle3.pt.hidden` 直接 `ENOMEM`，其余 5 个进程被 OOM
killer 在 epoch 1-2 中途 `Killed`。此外 `run_train.sh` 没给 `train_multimax.py`
传 `--mmap_cache`，导致 MultiMax 把 100-200GB 的 mid 缓存整个读进 RAM。

修复：
- `cache_io.py` 新增 `_load_bundle()`，mmap 加载强制 `MAP_SHARED`（文件支撑、
  不计 commit、干净页可回收）。已实测 608GB 缓存单进程可正常加载。
- `run_train.sh` 给 MultiMax 补 `--mmap_cache`；并加 `run_if_missing` 幂等跳过
  （已有 `final_results.json` 的方法不重跑）。
- `cot_local/rerun_training.sh`：6 单元监督器，最多 3 轮重试，OOM 后可续跑。

**② 标注解析 bug（深层问题，导致数字无效）**：`label_traces.py` 开启 thinking，
但 vLLM 返回的截断输出里 0/11616 含 `</think>`；回退逻辑把整段
"Thinking Process:" 长文当答案，用 `"yes" in answer` / `"no" in answer` 子串匹配，
且 `"yes"` 先判 → 2622 条截断样本中 2560 条被误标为正例。
实测 `cache_cot_repetitive_4b.pt` 的 5000 条训练样本里 **2154 条是这样的伪正例**
（正类 86% 是噪声）；`cot_branched` 也有 2328 条截断样本（2317 条被误标 branched）。
证据：同一个 last-token 逻辑回归，用原标签 70.5%，只用干净短答案标签 78.6%
（训练样本还少 7 倍）。这解释了为何 SP/MLP 在 Table 3 只有 ~69-72%。

**③ 更根本的限制：教师模型语料缺少目标现象**。尝试修复时发现：
- 强制 `enable_thinking=False`（空 think 块）会让裁判几乎全答 no/linear
  （随机 300 条：repetitive 0 正、branched 2 正），**不可用**；
- 让 thinking 跑完再取 `</think>` 后裁决虽能得到真实 yes/no，但 decode 仅
  ~400 tok/s，重标 4300 条约需 12h，**代价过高**；
- 真实（thinking-on、正确解析）的干净裁决里，repetitive 正率仅 **4.7%**
  （422/8994），branched **28.5%**（2622/9212）。本地 122B-A10B 是最强开源
  MoE，生成的 CoT 几乎没有"无意义重复思考"，而论文用 Qwen3.5-Plus 生成、
  正率约 25%。**这是教师模型差异带来的数据分布差异，不是脚本能修的**；
  要严格复现必须用 Qwen3.5-Plus 生成轨迹（本环境无 API）。

### 9.2 当前采用的方案（用户选择"修复并重跑全部"）

放弃重标注（慢/偏），改为**只保留真实短裁决**：
- `cot_local/make_fixed_labels.py` 从原标注文件筛出 `judge_output.raw` 恰为单个
  裁决词的记录，写成 `*_labeled_<task>_fixed_all.jsonl`：
  repetitive 8994 条（422 正 / 8572 负），branched 9212 条（2622 正 / 6590 负）。
- `precompute_hidden_states_cot.py::load_cot_data` 改为**稳健平衡采样**：先留
  平衡 held-out 池，再按可用正例数构造尽可能大的平衡训练集（原来固定切 2500
  正例会在正例不足时产出全负测试集）。dry-run：repetitive train=444/test=400，
  branched train=4844/test=400。
- `eval_split.py` 改为**分层平衡**划分 val/test，且不再从 train 借样本。
- `run_precompute_one.sh` 改指向 `*_fixed_all.jsonl`。
- `cot_local/rerun_all.sh`：删旧 cot 缓存/结果 → 6 路重算 precompute →
  `rerun_training.sh` 训练 → `compile_results.py` 汇总。原始（带 bug）数字备份在
  `cot_local/RESULTS_original_buggy.txt`。

**预期偏差（须写进报告）**：Table 3 训练集只有 444 条（论文 ~4950），数字仅作
趋势参考、方差大；Table 4 训练集 4844 条，量级接近论文。两表数值与论文的差异
首要来源是教师模型（122B-A10B-Int4）生成的轨迹分布，而非实现错误。

### 9.3 续跑操作

`bash cot_local/status.sh`／`cat cot_local/RESULTS.txt`／`tail -20 cot_local/rerun_all_main.log`／
`ls -la cot_local/repre_*.log`（precompute 进度）。监督器自己会接力训练和汇总，
结束后看 `cot_local/RESULTS.txt`。GPU 0-1 是他人进程，勿动；教师 vLLM 已停，
2-7 全部给 precompute/训练。

---

## 10. 最终结果（2026-11-18 20:38 完成）

`rerun_all.sh` 于 20:38 全部完成：6/6 precompute `PRECOMPUTE_ONE_DONE`、
30/30 `final_results.json`、训练日志 0 个 Traceback/Killed/OOM。缓存标签复核：
repetitive 444 train(222 正)/400 test(200 正)，branched ~4844 train(2422 正)/
400 test(200 正)，全部平衡。`cot_local/RESULTS.txt` 为下表，
`cot_local/RESULTS_original_buggy.txt` 为修复前（仅 6 格有效）的对照。

### Table 3 — CoT Repetition（复现值/论文值，%）

| 模型 | MLP | MM | SP-1 | SP-2 | SP-5 |
|---|---|---|---|---|---|
| Qwen3.5-4B | 79.5/87.5 | 77.5/87.0 | 76.5/88.0 | 75.5/87.5 | 82.5/89.0 |
| Qwen3.5-9B | 82.5/89.5 | 70.0/87.5 | 77.5/91.5 | 75.0/89.5 | 71.5/89.5 |
| MiniCPM4.1-8B | 78.5/83.5 | 64.5/82.5 | 68.5/86.0 | 66.5/85.0 | 64.0/87.0 |

### Table 4 — CoT Reasoning Strategy / branched（复现值/论文值，%）

| 模型 | MLP | MM | SP-1 | SP-2 | SP-5 |
|---|---|---|---|---|---|
| Qwen3.5-4B | 69.5/73.0 | **85.5/85.5** | 85.0/83.0 | 86.0/81.5 | 86.5/82.5 |
| Qwen3.5-9B | 70.5/76.5 | **86.5/87.0** | 87.5/84.5 | 86.5/83.0 | 86.0/83.5 |
| MiniCPM4.1-8B | 71.0/68.0 | **78.5/78.5** | **66.5/66.5** | 66.5/65.5 | 58.5/64.5 |

### 与论文的逐项对比（compile_results.py 内置论文值）

| 指标 | Table 3 | Table 4 |
|---|---|---|
| 平均绝对误差 | 13.37 pt | **2.63 pt** |
| 平均有符号误差 | −13.37（全部偏低） | +0.50 |
| 误差 ≤0.5pt 的格 | 0/15 | **4/15** |
| 误差 ≤3pt 的格 | 0/15 | **9/15** |
| 最准的格 | — | 4B/9B/MCPM 的 MM 全部 ≤0.5pt；MCPM SP-1/SP-2 也 ≤1pt |

**Table 4 复现成功**：MultiMax 三个模型几乎逐点命中（85.5/85.5、86.5/87.0、
78.5/78.5），MiniCPM 的 SP-1/SP-2 命中，Qwen 的 SP 略高于论文 2-4pt。
论文的核心趋势（MM/SP 显著优于 MLP；MiniCPM 在该任务整体偏弱）完全成立：
4B MM−MLP = +16.0pt（论文 +12.5），9B = +16.0pt（论文 +10.5）。

**Table 3 仍整体偏低 ~13pt**，且 15 格全部低于论文：根因是 §9.1③ 的数据
分布差异——干净裁决下正例只有 422 条，平衡后训练集仅 444 条（论文 ~4950），
探针欠训练、方差大；MM/SP 在 MiniCPM 上掉到 64-68 尤其明显。相对趋势仍可辨：
4B 上 SP-5 最高（82.5）与论文 SP-5 最高（89.0）一致。相比修复前的
69.0/66.5/71.5，干净标签已把 4B SP-1/SP-2、9B SP-1 提升 5-8pt。

### 结论

1. 流水线实现与论文协议一致（平衡 200/200 val/test、mid_layer=16、SP/MLP/MM
   训练配置），OOM 与标注 bug 均已修复并经完整性校验；
2. Table 4 数值级复现成功，方法排序趋势也对；
3. Table 3 的偏差来自**教师模型语料**（本地 122B-A10B-Int4 生成的 CoT 几乎
   不含"无意义重复思考"，正率 4.7% vs 论文约 25%），导致可用正例与训练集规模
   远小于论文；这不是实现错误，也无法在不换教师模型的前提下消除；
4. 若要严格复现 Table 3，需用 Qwen3.5-Plus 生成轨迹（本环境无 API），或改用
   更弱的生成模型诱发重复思考，然后重跑 ②→⑥（`gen_traces.py` →
   `label_traces.py`（已修严格解析）→ `run_precompute_one.sh` → `run_train.sh`）。

### 本次改动文件

- 修复：`fast_softprompt/cache_io.py`（MAP_SHARED）、`eval_split.py`（分层平衡）、
  `precompute_hidden_states_cot.py`（稳健平衡采样）、`cot_local/label_traces.py`
  （严格裁决解析）、`cot_local/run_train.sh`（MultiMax --mmap_cache + 幂等跳过）、
  `cot_local/run_precompute_one.sh`（指向 `*_fixed_all.jsonl`）。
- 新增：`cot_local/make_fixed_labels.py`、`rerun_training.sh`、`rerun_all.sh`、
  `restart_vllm.sh`；数据与结果：`*_fixed_all.jsonl`、`RESULTS.txt`、
  `RESULTS_original_buggy.txt`、30 个 `final_results.json`。
