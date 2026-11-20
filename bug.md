# 分支代码审查：硬性 Bug 清单

> 审查日期：2026-11-20
> 审查范围：`main` 之外的全部分支
> - `CoT推理轨迹复现`
> - `deepseek-v4.1-flash-1017-cal-time`
> - `new_benchmark`
>
> 审查方法：以各分支与 `main` 的 merge-base（`613e6ff`）为基准做 `git diff`；对新增/修改的 Python 文件跑 `pyflakes` 静态分析 + 全量 `compileall` 语法检查；逐文件人工核对字段、函数签名、路径与数据流；并用项目自带 conda 环境（`/home/suzeqin/miniconda3/envs/sp/bin/python`，torch 2.8.0）验证关键运行时行为。
>
> 说明：本报告只记录**确定性/接近确定性的硬性 bug**（会崩溃、或必然产生错误结果/错误数据），以及少量触发条件明确的"较可能"问题。纯风格、未使用导入/变量等不计入。所有结论均以各分支当时的代码为准，行号为该分支文件内行号。

## 结论摘要

| # | 分支 | 严重度 | 位置 | 一句话 |
|---|------|--------|------|--------|
| 1 | `new_benchmark` | **致命** | `fast_softprompt/precompute_hidden_states.py:127,157,233` | 全局改名引入未定义变量 `add_special`，3 个 task 必崩（NameError） |
| 2 | `new_benchmark` | **严重** | `fast_softprompt/train_mlp.py:199`、`train_multimax.py:264`、`train_eagle3_minicpm.py:372` | 误删 `load_state_dict(best_state)`：`best.pt` 存成最后一个 epoch 的权重，且最终测试指标用错模型 |
| 3 | `new_benchmark` | 中 | `specbench/spec_decode.py:395-396` | `Eagle3SpecEngine.eos_ids` 取的是 `GenerationConfig` 对象，EOS 永不生效，生成跑满 `max_new_tokens` |
| 4 | `deepseek-v4.1-flash-1017-cal-time` | 中 | `latency/merge_latency.py:229-231` | `pre_ev_min` 仅在 `if pre_dev:` 内赋值，格式化 f-string 时可能 `UnboundLocalError`/`TypeError` 崩溃 |
| 5 | `new_benchmark` | 低（触发明确） | `specbench/run_bench.py:244-247` | 全部样本失败时 `None` 被 `:.3f` 格式化，`TypeError` 崩溃 |
| 6 | `new_benchmark` | 低（依赖 CWD） | `specbench/probes.py:190-202` ↔ `specbench/run_bench.py:154` | 探针拿到相对 `base/head` 路径、引擎拿到绝对路径，从非仓库根启动会加载失败 |

`CoT推理轨迹复现` 分支未发现确定性硬性 bug（详见末尾说明）。

---

## 分支：`new_benchmark`

### Bug 1（致命）未定义变量 `add_special`，`--task pii/longsafety/nemotron_safety` 必然崩溃

- 位置：`fast_softprompt/precompute_hidden_states.py:127`、`:157`、`:233`
- 涉及函数：`load_pii_data`、`load_nemotron_safety_data`、`load_longsafety_data`
- 现象：运行下列任意命令立即抛 `NameError: name 'add_special' is not defined`：
  ```
  python fast_softprompt/precompute_hidden_states.py --task pii ...
  python fast_softprompt/precompute_hidden_states.py --task nemotron_safety ...
  python fast_softprompt/precompute_hidden_states.py --task longsafety ...
  ```
- 根因：该分支为新增的 `load_new_benchmark_data(..., add_special=True)` 增加了一个 `add_special` 形参，同时把文件里原有的
  `tokenizer.encode(r["text"], add_special_tokens=True)` 一并全局替换成了
  `tokenizer.encode(r["text"], add_special_tokens=add_special)`（见 `git diff main...new_benchmark` 该文件的 3 处 hunk）。
  但 `load_pii_data` / `load_nemotron_safety_data` / `load_longsafety_data` 的函数签名里**并没有** `add_special` 参数，
  模块级也没有定义该变量（全文件仅 7 处 `add_special` 出现，全为使用）。`main()` 中这三个 task 的分支（`:305`、`:316`、`:326`）仍会调用它们。
- 对比：`main` 中这三处都是 `add_special_tokens=True`，可正常工作；属于分支引入的回归。
- 修复建议：把这三处的 `add_special` 改回 `True`（或显式给三个 loader 增加 `add_special=True` 形参）。
  `load_new_benchmark_data`（`:199`）里的 `add_special` 是其形参，保持不变即可。

### Bug 2（严重）三个训练脚本误删 `load_state_dict(best_state)`，导致 `best.pt` 权重错误 + 最终指标错误

- 位置：
  - `fast_softprompt/train_mlp.py:199-206`
  - `fast_softprompt/train_multimax.py:264-271`
  - `fast_softprompt/train_eagle3_minicpm.py:372-379`
- 现象（两个同时存在的确定性错误）：
  1. **`best.pt` 存的是最后一个 epoch 的权重，却标着 best-epoch 的元数据**：保存用的是
     `classifier.state_dict()`（当前/最后 epoch），但 `"epoch": best_epoch, "val_acc": best_eval_acc`
     来自 best epoch。任何加载该 checkpoint 的下游代码（`specbench/probes.py`、`specbench/eval_detection.py`、`run_bench.py`、`eval_asr.py`）拿到的都是错误权重。
  2. **最终 test/val 指标用错模型**：原来 `main` 在最终评测前会 `classifier.load_state_dict(best_state)`
     还原验证集最优模型，再算 `test_acc` / `vA` 等并写入 `final_results.json`（见 `train_mlp.py:208-228`）。
     该分支把这行**删除**了（替换成 `torch.save`），因此 `final_results.json` 的 `metrics.accuracy` / `metrics.f1`
     是最后 epoch 的结果，而非 best epoch，`new_benchmark_repro.py:verify()` 读取的正是这些值。
- 根因：分支为了产出探针 checkpoint（`best.pt`）修改这三个脚本时，把 `if best_epoch > 0:` 块里的
  `load_state_dict(best_state)` 直接替换成了保存逻辑，而不是"先还原、再保存、再评测"。`best_state` 仍在循环里被计算
  （`train_mlp.py:194` 等）但已完全未使用（pyflakes 亦报 `assigned but never used`）。
  同分支的 `train_softprompt.py:818,877` 保留了 `load_state_dict(best_state, strict=False)` 且保存的是 `best_state`，
  写法正确 —— 反证这三处是编辑失误。
- 修复建议：恢复 `best_state` 还原，并且 `best.pt` 应保存 `best_state` 而非当前 `state_dict()`，例如：
  ```python
  if best_epoch > 0:
      classifier.load_state_dict(best_state)          # 先还原最优模型
      if args.output_dir:
          ... torch.save({..., "state_dict": best_state}, .../best.pt)
  tp, tg, test_acc, ... = eval_on(test_features, test_labels)   # 再评测
  ```

### Bug 3（中）`Eagle3SpecEngine` 的 EOS 集合构造错误，生成永不停止

- 位置：`specbench/spec_decode.py:395-396`
  ```python
  eos = getattr(self.model, "generation_config", None)          # 拿到的是 GenerationConfig 对象
  self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None
  ```
- 现象：`eos_ids` 变成 `{GenerationConfig(...)}`（一个装着配置对象的集合）。`SpecEngineBase.generate()`
  （`:304`）用 `self.committed[-1] in self.eos_ids` 判断，其中 `committed[-1]` 是 int token id，
  `int in {GenerationConfig}` 恒为 `False` → **EOS 永不触发，生成一直跑到 `max_new_tokens`**。
  已在项目 conda 环境验证：`GenerationConfig` 可哈希（不会在构造时报错），且 `1 in eos_ids == False`。
- 对比：同一文件 `QwenSpecEngine`（`:333-335`）和 `GreedyEngine`（`:503-505`）都正确取了 `.eos_token_id`：
  ```python
  eos = gc.eos_token_id ...
  ```
  唯独 `Eagle3SpecEngine` 漏了 `.eos_token_id`，属复制粘贴遗漏。
- 修复建议：
  ```python
  gc = getattr(self.model, "generation_config", None)
  eos = gc.eos_token_id if gc is not None else None
  self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None
  ```
- 影响面：`run_bench.py` / `eval_asr.py` 中所有 `minicpm4.1-8b`（Eagle3）配置。

### Bug 5（低，触发条件明确）`run_bench.py` 全部样本失败时打印崩溃

- 位置：`specbench/run_bench.py:244-247`
- 现象：当某个 config 下所有 example 都进入 `except`（:207 的 `continue`，例如引擎报错/全部 OOM），`per_example` 为空，
  于是 `acceptance_rate_mean` / `acceptance_length_mean` 为 `None`；紧接的
  `f"acc={config_summary['acceptance_rate_mean']:.3f}"`、`tau={...:.2f}` 会对 `None` 做浮点格式化 → `TypeError`。
  本应优雅输出"0 个成功样本"，却在打印处二次崩溃。
- 修复建议：打印前判空，或用 `fmt` 类函数把 `None` 显示为占位符。

### Bug 6（低，依赖工作目录）探针与引擎的模型路径一致性

- 位置：`specbench/probes.py:190-202`（`load_probe`）、`specbench/run_bench.py:154-162`
- 现象：`run_bench.py` 构造引擎时用 `os.path.join(ROOT, spec["base"])` / `spec["head"]`（**绝对**路径），
  但传给 `load_probe` 的 `spec` 里 `base/head` 仍是 `"models/Qwen3.5-4B"` 等**相对**路径；
  `load_probe` 里只有 `probes_root` 被 `os.path.join(ROOT, ...)` 绝对化，`base`、`head` 原样传给
  `SPProbeQwen` / `SPProbeEagle3`，后者会用它去 `from_mtp_checkpoint` / 加载 tokenizer。
  一旦脚本不是从仓库根目录启动（未按 README 用法），探针加载会找不到模型目录而失败。
- 修复建议：在 `load_probe` 里对 `base`/`head` 也用 `ROOT` 绝对化，或直接传入引擎已用的绝对 `spec`。

---

## 分支：`deepseek-v4.1-flash-1017-cal-time`

### Bug 4（中）`merge_latency.py` 中 `pre_ev_min` 可能未定义 / 为 `None`，f-string 崩溃

- 位置：`latency/merge_latency.py:226-231`
  ```python
  pre_ev = next((r["prefill_event_ms"] for ...), None)          # :227 算出来却从未使用
  dec_dev = next((r["decode_device_ms"] for ...), None)
  if pre_dev:                                                   # :229
      pre_ev_min = next((r["prefill_event_min_ms"] for ...), None)   # 仅在此分支内赋值
  lines.append(f"Reference: base prefill **{pre_ev_min:.2f} ms** ...")   # :231 无条件使用
  ```
- 现象（三种失败模式，均为确定性代码路径问题）：
  1. **所有 run 的 `pre_dev` 都为假**：`pre_ev_min` 在本函数作用域内从未被赋值 → `UnboundLocalError`（NameError 子类）崩溃。
     典型触发：全部结果用 `bench_latency.py --no_profile` 生成时，`device_time_us_by_length` / `device_time_us` 不存在，
     `prefill_device_ms`（即 `pre_dev`）为 `None`。
  2. **首个 run 有 `pre_dev`、后续某个 run 没有**：由于是同一函数作用域的循环，`pre_ev_min` 会**残留上一个模型/任务的值**，
     静默输出错误的"base prefill 参考值"（不易察觉的正确性 bug）。
  3. **`pre_dev` 为真但没有 `prefill_event_min_ms`**：`pre_ev_min = None`，`f"{None:.2f}"` → `TypeError`。
- 附注：`:227` 的 `pre_ev`（`prefill_event_ms`）本应作为兜底值使用，却被弃用（pyflakes 也报 `unused`）。
- 修复建议：把 `pre_ev_min` 移到 `if` 外无条件赋值，并给 `None` 兜底，例如
  ```python
  pre_ev_min = next((r["prefill_event_min_ms"] for r in by_method.values()
                     if r.get("prefill_event_min_ms")), None) or pre_ev
  ... f"**{ms(pre_ev_min)} ms** ..."   # 用能处理 None 的格式化函数
  ```

### 该分支其余文件
`latency/common.py`、`bench_latency.py`、`profile_scaling.py`、`repair_profile.py`、
`latency/run_all_latency.sh`、`run_scaling.sh`，以及 `train_eagle3_minicpm.py` / `minicpm_eagle3_wrapper.py`
的改动，经逐文件核对未发现确定性硬性 bug：
- `pyflakes` 在 `common.py:391/398/404` 报的 `undefined name a/b/dst/src/x/w` 为**误报** —— 这些变量在 lambda 前已定义、
  在 lambda 后被 `del`，是 pyflakes 对「lambda 闭包 + del」的已知误判，已用最小脚本复现验证，代码本身合法。
- `minicpm_eagle3_wrapper.py` 新增的 `skip_lm_head` / `past_key_value` 与 `train_eagle3_minicpm.py:102` 的调用签名一致，
  且都给了默认值，向后兼容。

---

## 分支：`CoT推理轨迹复现`

未发现确定性硬性 bug。该分支本质上是一个**修复分支**（提交 `3c51ae5` 修 OOM 与子串解析 bug，`ef2371e` 清理脏数据），
其改动方向是修正而非引入问题：

- `cache_io._load_bundle` 用 `torch.utils.serialization.config.load.mmap_flags = mmap.MAP_SHARED` 规避
  `MAP_PRIVATE` 的 commit 计费 OOM —— 已确认在 torch 2.8.0 中该属性存在（默认 `2`=MAP_PRIVATE），写法有效。
- `label_generated_simpleqa_...py` 与 `label_traces.py` 把子串匹配改为「短答案 + 首词精确匹配」，修掉误判。
- `eval_split.py` 改为分层划分、`precompute_hidden_states_cot.py` 改为稳健均衡采样。
- `cot_local/` 下的 `gen_traces.py` / `label_traces.py` / `compile_results.py` 及各 `.sh`：
  已核对 monkeypatch 作用域（`g.OpenAI` / `g.generate_one` 等对模块全局生效）、
  `load_samples` 字段、输出文件名（`*_labeled_<task>_fixed_all.jsonl`）与 `run_precompute_one.sh` 的 `DATA` 路径一致、
  缓存命名（`*_mid.pt` / `_mcpm_eagle3.pt`）与 `run_train.sh` 一致，未发现崩溃或错误数据问题。

> 一个可选的健壮性观察（非硬性 bug）：`benchmark_gpt_zeroshot.py` 由 `torch.load(..., mmap=True)` 改为
> `cache_io.load_cache(..., mmap=True)` 后，对 `mmap_v1` 格式缓存会强制要求 `.pt.hidden` 伴生文件存在，
> 而它本身只用到 `token_ids/labels/model` 等元数据。当前仓库的 `.hidden` 文件齐全，可正常工作；
> 若缓存的 `.hidden` 被单独清理，会抛 `FileNotFoundError`。属环境依赖，不计入硬性 bug。

---

## 修复优先级建议

1. **Bug 1** —— 直接阻断 3 个 precompute task，优先修。
2. **Bug 2** —— 直接污染 `best.pt` 与全部检测/ASR/bench 结果与 `final_results.json`，影响面最大，且会静默产出错误数据，务必优先。
3. **Bug 3** —— 影响所有 MiniCPM 生成的长度与语义。
4. **Bug 4** —— 影响报告生成（尤其 `--no_profile` 场景）。
5. Bug 5 / 6 —— 边界/路径问题，可随后处理。
