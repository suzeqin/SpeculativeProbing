# Inference-time overhead of Speculative Probing and the probe baselines

The paper reports only *asymptotic* added cost (Table 6: `O(d²+nd)` for SP-1,
`O(kd²+knd)` for SP-k, `O(ndd')` for MultiMax) and the training hardware
(Appendix B: 2×RTX PRO 6000). It never reports measured latency for SP-1 /
SP-2 / SP-5, MLP or MultiMax. This directory adds that measurement.

Everything here reuses the **exact modules that produced the accuracy numbers**
in `results/`:

| component | source |
|---|---|
| base models | `models/Qwen3.5-4B`, `models/Qwen3.5-9B`, `models/MiniCPM4.1-8B` |
| MTP head | `models/Qwen3.5-{4B,9B}-mtp` via `qwen35_mtp.Qwen3_5MTP` |
| Eagle3 head | `models/MiniCPM4.1-8B-Eagle3/...` via `minicpm_eagle3_wrapper.py` |
| SP classifier | `fast_softprompt/train_softprompt.RecursiveSoftPromptClassifier` (Qwen), `train_eagle3_minicpm.Eagle3SoftPromptClassifier` (MiniCPM) |
| MLP probe | `fast_softprompt/train_mlp.MLPClassifier` (hidden 1024, as reproduced) |
| MultiMax | `fast_softprompt/train_multimax.MultiMaxClassifier` (H=10, width 100) |
| data split | identical `eval_split.make_val_test_split` indices (seed 1337) |
| features | the reproduced `cache_*.pt` hidden states (mmap), so no re-run of the base model is needed to get inputs |

## What is measured

For every test example (batch size 1, bf16, no `torch.compile`):

| name | definition |
|---|---|
| `base_prefill` | base LM forward over the prompt with `output_hidden_states=True, use_cache=True` — the pass that produces the features the probes consume |
| `base_prefill_nocache` | same without hidden-state retention / KV cache |
| `base_decode` | one cached autoregressive decoding step (per-token cost) |
| `mlp` | 2-layer MLP on the last-token last-layer hidden state |
| `mm` | MultiMax on the mid-layer hidden states of all positions (layers 8 / 16 / 16) |
| `sp{k}` | **standalone**: the reproduced SP forward (draft head over the whole sequence, `k` recursive/appended soft prompts) |
| `sp{k}_inc` | **KV-cached deployment**: the draft head's prefix KV cache is already resident (speculative decoding); only the `k` soft-prompt positions are evaluated (`k` single-token draft-head steps + the linear head), matching Eq. (1)–(3) of the paper |
| `sp{k}_core` | MiniCPM only: `sp{k}` without the draft-vocabulary `lm_head` + `d2t` scatter, which the classifier never reads |

`sp{k}_inc` is validated to produce the same logits as the reference forward
(up to bf16 kernel numerics; max |Δlogit| ≤ 0.05).

## Timing methodology

* **GPU time** (headline): summed CUDA kernel durations of one call, via
  `torch.profiler` CUPTI activity. This is the added device work and the
  quantity that matters when the probe runs asynchronously in a serving loop.
* **Eager latency**: per-call CUDA-event window (`start.record(); call;
  end.record()`), which includes Python launch gaps and in-forward
  synchronisation of the eager reference implementation.
* Repeated `--reps` times per example; **min** is used to suppress bursts of
  interference from the other tenants sharing the GPU, median/p90 are kept for
  the observed distribution.
* 32 MiB L2 flush between timed calls (features are produced by a multi-ms
  prefill, so they are not L2-resident by the time the probe runs).
* Methods are interleaved (round-robin over trials) inside each probe-profile
  bucket so a co-tenant's contention burst cannot systematically favour one
  method.
* Reference microbenchmarks (dense 4096³ bf16 matmul, 1 GiB copy, 8192×4096
  GEMV) are measured immediately before and after each run; they are stored in
  the JSON (`calibration_start/end`). On the RTX A6000 used here they read
  ~125 TFLOP/s and ~680 GB/s, i.e. the GPU is shared but not throttled.

## Reproduce

```bash
# one (model, task)
python latency/bench_latency.py --model qwen3.5-4b --task verih \
  --methods mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc \
  --out latency/results/qwen3.5-4b_verih.json

# all six (model, task) pairs + tables
bash latency/run_all_latency.sh 6

# rebuild the tables from existing JSONs
python latency/merge_latency.py
```

Outputs: `latency/results/<model>_<task>.json` (raw per-example samples),
`latency/results/latency_summary.csv`, `latency/results/LATENCY_TABLE.md`,
and the human-readable `LATENCY_REPORT.md` in the repository root.

Environment: `conda activate sp` (PyTorch 2.8.0+cu126, Transformers 5.20.2).
`flash-linear-attention` 0.3.2 is installed in the `sp` environment so the
Qwen3.5 linear-attention layers use the optimized `fla` kernels instead of the
reference PyTorch fallback; `causal-conv1d` is **not** installed, so the
depthwise conv in those layers uses the reference implementation (this only
affects the base-model reference, not any probe).
