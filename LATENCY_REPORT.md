# Measured inference-time overhead of Speculative Probing and the probe baselines

**Question.** The paper *Speculative Probing: LLM Monitoring at Speculative-Decoding
Cost* (Zhang, Zhang, Shmatikov) argues that its probes are essentially free at
inference time, but it only supports that claim with **asymptotic** costs
(Table 6: `MLP O(dh)`, `MULTIMAX O(n·d·d′)`, `SP-1 O(d²+nd)`, `SP-5 O(kd²+knd)`)
and an **ordinal** cost axis in Figure 3. Appendix B reports only training
hardware (2× RTX PRO 6000) and training wall-clock. No measured inference
latency for SP-1 / SP-2 / SP-5 / MLP / MultiMax appears anywhere in the paper
or appendix.

**What this report adds.** Wall-clock measurements of every probe on the exact
reproduction checkpoints and data used in `REPRODUCTION_REPORT.md`, for
Qwen3.5-4B, Qwen3.5-9B and MiniCPM4.1-8B on VerIH and Nemotron-Safety.

- Code: [`latency/`](latency/README.md) — `bench_latency.py`,
  `profile_scaling.py`, `repair_profile.py`, `merge_latency.py`,
  `run_all_latency.sh`.
- Raw data: `latency/results/<model>_<task>.json` (per-example samples,
  per-length buckets, calibration), `latency/results/scaling_*.json`.
- Generated tables: [`latency/results/LATENCY_TABLE.md`](latency/results/LATENCY_TABLE.md),
  [`latency/results/latency_summary.csv`](latency/results/latency_summary.csv).

---

## 1. What is measured

Every probe is timed at **batch size 1, bf16, PyTorch eager, no CUDA graphs /
`torch.compile`**, on the test split that produced the reported accuracies
(same `eval_split.make_val_test_split` indices, seed 1337). The probe modules
are imported from the reproduction code, not re-implemented:

| probe | module | inputs |
|---|---|---|
| MLP | `train_mlp.MLPClassifier` (`hidden=1024`, dropout 0.1) | last-token last-layer hidden state |
| MultiMax | `train_multimax.MultiMaxClassifier` (H=10, width 100) | mid-layer hidden states of **all** positions |
| SP-k | `train_softprompt.RecursiveSoftPromptClassifier` (Qwen MTP), `train_eagle3_minicpm.Eagle3SoftPromptClassifier` (MiniCPM Eagle3) | last-layer hidden states + token ids |

Two SP variants are measured, because they correspond to two different
deployment points:

* **`sp{k}` — standalone.** The forward pass exactly as the reproduction runs
  it: the draft head (MTP / Eagle3) is executed over the whole sequence, then
  `k` recursive / appended soft-prompt positions are evaluated.
* **`sp{k}_inc` — KV-cached.** The deployment assumed by §3.3 of the paper:
  speculative decoding has already run the draft head over the verified prefix,
  so its KV cache is resident. Speculative Probing then only evaluates the `k`
  soft-prompt positions (`k` single-token draft-head steps + the linear head).
  This is the `O(k·n·d_h) attention + O(k·d²) feed-forward` term of Eq. (1)–(3).
  It is validated to produce the same logits as the reference forward
  (max |Δlogit| ≤ 0.05 in bf16, i.e. < 0.02 in sigmoid probability).

`sp{k}_core` (MiniCPM only) additionally removes the frozen draft-vocabulary
`lm_head` + `d2t` scatter: the classifier reads only the Eagle3 hidden state, so
that 4096→32000 projection is dead work. The reproduction's forward performs it
anyway; we report both.

**Reference quantities** measured on the same examples and in the same time
window:

* `base_prefill` — base LM forward over the prompt, `output_hidden_states=True,
  use_cache=True` (the pass that produces the features every probe consumes).
* `base_decode` — one cached autoregressive decode step (per-token cost of
  continuing the generation that the monitor would watch).

**Timing method.** Per-call CUDA-event windows (`start.record(); call;
end.record()`), 5 repetitions per example, 32 MiB L2 flush between calls.
Reported **latency** is the median over test examples of the per-example
*minimum* over repetitions: the minimum suppresses bursts of interference from
the other tenants that share the GPU; the median over examples makes it a
per-request cost. **GPU-kernel time** is the summed CUDA kernel duration of one
call from `torch.profiler` (median-length example, median over 5–7 trials).
Because the paper's claim is about a probe running inside a serving loop, both
matter: latency is what an eager caller observes, GPU time is the added device
work that an optimized engine would pay.

**Accuracy** comes from the completed reproduction (`results/*/*/final_results.json`);
the timing uses the same module shapes with fixed-seed weights (probe latency
depends on shapes and `k`, not on weight values; the soft-prompt/head weights
are unchanged in shape and execution path).

---

## 2. Headline result

**Measured per-call latency** (ms), base prefill as reference, accuracy of the
same configuration. Full per-model tables are in
[`latency/results/LATENCY_TABLE.md`](latency/results/LATENCY_TABLE.md).

| Model | Task | prefill | MLP | MultiMax | SP-1 | SP-2 | SP-5 | SP-1 KV | SP-2 KV | SP-5 KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 60.3 | 0.034 | 0.068 | 3.18 | 8.46 | 24.1 | 0.68 | 1.32 | 5.59 |
| Qwen3.5-4B | nemotron | 47.8 | 0.034 | 0.048 | 3.03 | 8.31 | 24.0 | 0.67 | 1.31 | 5.59 |
| Qwen3.5-9B | verih | 76.6 | 0.044 | 0.078 | 3.58 | 9.19 | 25.9 | 1.01 | 4.30 | 9.58 |
| Qwen3.5-9B | nemotron | 41.1 | 0.043 | 0.11 | 1.60 | 3.09 | 7.26 | 1.13 | 2.10 | 5.03 |
| MiniCPM4.1-8B | verih | 108.0 | 0.044 | 0.080 | 6.38 | 6.37 | 6.41 | 1.08 | 4.46 | 9.93 |
| MiniCPM4.1-8B | nemotron | 101.2 | 0.043 | 0.058 | 6.17 | 6.21 | 6.21 | 1.07 | 4.44 | 9.91 |

**Overhead relative to the base prefill** (same windows; sum of per-example
latencies):

| Model | Task | MLP | MultiMax | SP-1 | SP-2 | SP-5 | SP-1 KV | SP-2 KV | SP-5 KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 0.04% | 0.08% | 4.65% | 10.70% | 27.14% | 0.83% | 1.62% | 6.60% |
| Qwen3.5-4B | nemotron | 0.06% | 0.09% | 5.35% | 13.61% | 37.10% | 1.14% | 2.65% | 9.84% |
| Qwen3.5-9B | verih | 0.04% | 0.07% | 4.16% | 8.29% | 19.87% | 0.85% | 3.24% | 7.19% |
| Qwen3.5-9B | nemotron | 0.09% | 0.22% | 3.80% | 6.89% | 15.66% | 2.38% | 4.48% | 10.77% |
| MiniCPM4.1-8B | verih | 0.03% | 0.06% | 6.09% | 6.00% | 6.30% | 0.75% | 2.76% | 5.88% |
| MiniCPM4.1-8B | nemotron | 0.04% | 0.06% | 6.21% | 6.41% | 6.32% | 0.93% | 3.44% | 7.63% |

**Accuracy** (reproduced, %):

| Model | Task | MLP | MultiMax | SP-1 | SP-2 | SP-5 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 78.0 | 90.7 | 87.0 | 89.5 | 92.6 |
| Qwen3.5-4B | nemotron | 75.2 | 78.5 | 78.0 | 76.2 | 77.0 |
| Qwen3.5-9B | verih | 79.9 | 92.3 | 91.3 | 89.8 | 91.3 |
| Qwen3.5-9B | nemotron | 75.0 | 79.8 | 75.8 | 79.2 | 77.8 |
| MiniCPM4.1-8B | verih | 64.7 | 78.3 | 76.8 | 76.2 | 76.8 |
| MiniCPM4.1-8B | nemotron | 63.8 | 69.9 | 65.1 | 64.5 | 63.8 |

### Take-aways

1. **MLP and MultiMax are free** — 0.03–0.11 ms, i.e. **0.03–0.22 % of the
   prefill**, on every model/task. MultiMax reads all `n` positions yet stays
   under 0.11 ms because the per-position MLP is only `d→100`; its 0.27–0.42 M
   parameters are a handful of tiny GEMMs.
2. **SP-1 in the speculative-decoding deployment costs ~0.7–1.1 ms**: one
   draft-head decode step, **0.8–2.4 % of the prefill**. That is the paper's
   "negligible overhead" claim, and it holds.
3. **SP-5 in the same deployment costs 5–10 ms** (five draft-head steps,
   5.9–10.8 % of the prefill). It is small compared with a full request that
   generates tokens, but it is not negligible in absolute terms; the paper's
   `O(kd²+knd)` is visible as an almost exactly linear cost in `k`
   (`≈0.6 ms`/step of GPU time for Qwen-4B, `≈1.0 ms`/step for Qwen-9B and
   MiniCPM Eagle3; measured latency per step is larger because of Python/launch
   overhead, see §4).
4. **The standalone path is much more expensive than the paper's asymptotic
   story suggests** — up to 24–26 ms for SP-5 on Qwen — because of two
   implementation effects, not compute: the full-sequence draft-head pass
   (1.0–4.5 ms GPU) and, for `k>1`, host→device copies of `position_ids` /
   `cache_position` inside every recursive step (§5). With a resident KV cache
   and precomputed positions, the same computation takes 5–10 ms.
5. **MiniCPM's Eagle3 probe is the most expensive per step** (0.7–1.0 ms for
   SP-1, ~2 ms/step for `k≥2` in the eager stack) and its classifier is much
   larger (20.5 K / 36.9 K / 86.0 K trainable parameters for k=1/2/5, because
   the Eagle3 head takes four input streams, each with its own soft prompt,
   versus 5.1 K / 7.7 K / 15.4 K for Qwen). Its standalone forward also spends
   ~2.7–2.9 ms on the unused draft-vocabulary `lm_head`+`d2t` projection;
   removing it (`sp{k}_core`) cuts latency from 6.4 ms to 3.6 ms (VerIH) and
   from 6.2 ms to 3.5 ms (Nemotron).

**Cross-model note.** For MiniCPM the eager base prefill is unusually slow
(108 ms measured vs 40 ms of GPU kernels), because its remote-code modeling
file has substantially more Python per layer. That lowers MiniCPM's relative
percentages in the table above; the GPU-relative table in §3 is the fairer
cross-model comparison (MiniCPM's SP-1 standalone is then 11.3 % of the
prefill, vs 1.8–4.1 % for Qwen).

---

## 3. GPU-kernel time (device work per call)

Summed CUDA kernel durations, median-length example. This is the quantity that
would remain in a fused/CUDA-graph serving engine.

| Model | Task | prefill | MLP | MultiMax | SP-1 | SP-2 | SP-5 | SP-1 KV | SP-2 KV | SP-5 KV† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 56.9 | 0.031 | 0.061 | 1.01 | 1.64 | 3.50 | 0.63 | 1.22 | 3.00 |
| Qwen3.5-4B | nemotron | 17.2 | 0.031 | 0.044 | 0.70 | 1.30 | 3.16 | 0.62 | 1.22 | 2.99 |
| Qwen3.5-9B | verih | 45.0 | 0.041 | 0.070 | 1.66 | 2.63 | 5.47 | 0.96 | 1.90 | 7.06† |
| Qwen3.5-9B | nemotron | 27.6 | 0.040 | 0.044 | 1.08 | 2.04 | 4.86 | 0.96 | 1.88 | 7.01† |
| MiniCPM4.1-8B | verih | 39.8 | 0.040 | 0.071 | 4.49 | 4.58 | 4.60 | 1.04 | 4.40† | 9.84† |
| MiniCPM4.1-8B | nemotron | 28.2 | 0.040 | 0.051 | 1.59 | 1.62 | 1.61 | 1.03 | 2.06 | 9.83† |

† CUPTI instrumentation inflates multi-step calls (the profiler serialises and
re-instruments each of the ~40 kernels per draft-head step). The measured-latency
table in §2 is the reliable one for `k≥2`; the GPU time of a single draft-head
step is 0.6 ms (Qwen-4B), 0.95 ms (Qwen-9B) and ~1.0 ms (MiniCPM), measured in
isolation.

Notes that follow from this table:

* **MLP/MM device work is 30–80 µs** and does not grow with model size; MM grows
  with `n` (45 µs at 10–50 tokens → 76–95 µs at 400–600 tokens).
* **SP standalone device time scales with `n`** (full-sequence draft-head pass):
  Qwen-4B SP-1 goes 0.69 → 1.01 → 2.90 ms for 41 → 183 → 653 tokens; Qwen SP-5
  adds `k−1` steps on top.
* **SP-1 KV-cached device time is flat in `n`** (0.60–0.63 ms for Qwen-4B,
  0.94–0.96 ms for Qwen-9B, 1.02–1.08 ms for MiniCPM): the step streams the
  draft head's weights and attends over a cached prefix, so it is independent
  of prompt length. This is exactly the `O(kd²)` term of the paper's Eq. (1)–(3).
* The prose above uses the probe-only scaling runs in
  `latency/results/scaling_*.json`; the table is reproduced in §A.

---

## 4. Cost model: SP-k is `k` draft-head steps

Per-example measured latency versus the number of soft prompts, KV-cached
(median-length example; the standalone path additionally pays the full-sequence
draft pass and the position-tensor copies of §5):

| Model | Task | SP-1 | SP-2 | SP-5 | per extra step | fraction of prefill / step |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 0.68 | 1.32 | 5.59 | 1.23 | 2.0 % |
| Qwen3.5-4B | nemotron | 0.67 | 1.31 | 5.59 | 1.23 | 2.6 % |
| Qwen3.5-9B | verih | 1.01 | 4.30 | 9.58 | 2.14 | 2.8 % |
| Qwen3.5-9B | nemotron | 1.13 | 2.10 | 5.03 | 0.98 | 2.4 % |
| MiniCPM4.1-8B | verih | 1.08 | 4.46 | 9.93 | 2.21 | 2.0 % |
| MiniCPM4.1-8B | nemotron | 1.07 | 4.44 | 9.91 | 2.21 | 2.2 % |

The step count is the dominant term; each step is one MTP/Eagle3 layer forward
(FC `2d→d`, one decoder layer, linear head). In GPU time a step is 0.59 ms
(Qwen-4B), 0.94 ms (Qwen-9B) and ~1.0 ms (MiniCPM Eagle3). The eager latency is
larger because every step also costs ~0.5–1.0 ms of Python/kernel-launch time:
the step streams 0.2–0.6 GB of draft-head weights (0.2 GB Qwen-4B, 0.44 GB
Qwen-9B, 0.6 GB MiniCPM), so at batch 1 it is memory-bound and launch-heavy.

---

## 5. The standalone path is dominated by host-side tensor creation

For `k>1`, `RecursiveSoftPromptClassifier.forward` creates a fresh
`position_ids` tensor of shape `[3,B,1]` and a `cache_position` tensor **inside
every recursive step**:

```python
pos3 = torch.tensor([[[pos_idx]]]).expand(3, batch_size, 1).to(self.device)
cache_pos = torch.tensor([pos_idx], device=self.device)
```

Each of these is a pageable host→device copy. On the shared GPU used here one
such copy costs **≈2.3 ms** (measured directly: the same MTP step costs
0.70 ms with precomputed tensors and 5.32 ms with the copies). Consequently the
reproduction's SP-5 inference path is **host-bound**, not GPU-bound:

| | GPU kernels (ms) | measured latency (ms) |
|---|---:|---:|
| Qwen-4B SP-1 (single-step path; no position tensors) | 1.01 | 3.18 |
| Qwen-4B SP-5 (4 recursive steps) | 3.50 | 24.14 |
| Qwen-9B SP-5 | 5.47 | 25.89 |

The incremental engine precomputes the position tensors once per example; it
runs the same arithmetic (validated logit-equivalent) at 5.59 ms / 9.58 ms for
SP-5. **A ~4-line change to the reference inference path (hoist the position
tensors out of the loop / reuse a `cache_position` int) would remove ~18–20 ms
of SP-5 latency on this hardware.** This does not affect the accuracy numbers in
`results/` (it is inference-only and numerically identical), but it matters for
anyone deploying the code as shipped.

---

## 6. Scaling with sequence length

Median per-example latency (ms) by length tercile (test-set terciles; the full
table with all eight probes and both tasks is in
[`LATENCY_TABLE.md`](latency/results/LATENCY_TABLE.md)):

| Model | Task | tokens | prefill | MLP | MM | SP-1 | SP-5 | SP-1 KV | SP-5 KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | verih | 56 | 45.1 | 0.034 | 0.052 | 3.06 | 24.0 | 0.67 | 5.56 |
| Qwen3.5-4B | verih | 132 | 58.2 | 0.034 | 0.068 | 3.21 | 24.2 | 0.68 | 5.59 |
| Qwen3.5-4B | verih | 422 | 119.6 | 0.035 | 0.079 | 4.16 | 25.1 | 0.69 | 5.62 |
| Qwen3.5-9B | verih | 56 | 67.4 | 0.043 | 0.055 | 3.46 | 25.8 | 1.00 | 9.56 |
| Qwen3.5-9B | verih | 132 | 76.7 | 0.044 | 0.078 | 3.59 | 25.9 | 1.01 | 9.59 |
| Qwen3.5-9B | verih | 422 | 189.4 | 0.044 | 0.097 | 7.82 | 29.7 | 1.02 | 9.61 |
| MiniCPM4.1-8B | verih | 61 | 100.2 | 0.043 | 0.057 | 6.20 | 6.24 | 1.07 | 9.90 |
| MiniCPM4.1-8B | verih | 154 | 107.2 | 0.044 | 0.080 | 6.41 | 6.51 | 1.08 | 9.90 |
| MiniCPM4.1-8B | verih | 486 | 215.5 | 0.044 | 0.10 | 14.02 | 14.09 | 1.11 | 10.01 |

* **MLP is constant** (single vector); **MultiMax grows slowly** with `n`
  (50 → 100 µs for 50 → 500 tokens).
* **SP standalone grows with `n`** (the draft head runs over the whole prompt);
  the growth is almost entirely the GPU part, since the Python overhead is
  constant.
* **SP KV-cached is flat in `n`** — the point of Eq. (1)–(3): the cached prefix
  is attended, not recomputed.
* The base prefill grows roughly linearly with `n`; hence the *relative* probe
  overhead is largest for short prompts (Nemotron: median 26–55 tokens).

---

## 7. End-to-end request overhead

Two realistic scenarios, using the measured latencies (the probe is additive in
this eager stack):

* **Prompt classification only** (VerIH / Nemotron style): a request pays
  `prefill + probe`. SP-1 KV adds 0.8–2.4 %, SP-5 KV adds 5.9–10.8 %, MLP/MM
  add < 0.25 %.
* **Monitoring during generation** of 128 tokens: the probe is
  `probe / (prefill + 128·decode)`. With the eager decode cost measured here
  (24–29 ms/token because a 4–9 B model decode at batch 1 launches ~4 500
  kernels), the probe is **0.02–0.3 %** of a 128-token generation. In a fused
  engine the decode would be faster, but the probe would be unchanged, so this
  fraction is a lower bound.

---

## 8. Caveats

1. **Shared GPU.** All eight GPUs of the host were at 98–100 % utilisation by
   other tenants during the measurements. Absolute times vary by up to ~1.8×
   between runs (e.g. the 9B/Nemotron run landed in a quiet window and is
   uniformly ~1.8× faster than the 9B/VerIH run). Within a run, probes and base
   prefill are measured in the same window and interleaved, so the **ratios**
   are stable; the reference kernels (4096³ bf16 matmul 124–128 TFLOP/s,
   8192×4096 GEMV 643–649 GB/s) are stored in every JSON's
   `calibration_start/end`.
2. **Eager PyTorch.** No `torch.compile`, CUDA graphs or fused kernels. The
   measured latency includes Python/launch overhead; the GPU-kernel column
   isolates device work. A production engine would sit between the two.
3. **`causal_conv1d` is not installed** in the `sp` environment, so Qwen3.5's
   depthwise convolution uses the reference PyTorch kernel (~5 % of the base
   prefill). This affects the reference, not the probes.
4. **MiniCPM needs an explicit `Cache`** for `use_cache=True` in its remote
   code; the reproduction's precompute used `use_cache=False`, so this is the
   first measurement of its prefill with a KV cache.
5. **Batch size 1.** Batching amortises launch overhead and changes the
   memory-bound/launch-bound balance; it is not measured here.
6. **Weights.** Probe weights are freshly initialised with a fixed seed; probe
   latency is determined by shapes and `k`, not by parameter values. Accuracy is
   taken from the completed reproduction.
7. **Reference implementation defines "standalone".** The standalone SP numbers
   include the reference forward's host-side position-tensor copies; §5 isolates
   that effect and shows the compute-only cost.

---

## 9. Reproduce

```bash
conda activate sp
cd /data1/suzeqin/SpeculativeProbing

# one (model, task): probes + base reference + calibration
python latency/bench_latency.py --model qwen3.5-4b --task verih \
  --methods mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc \
  --reps 5 --reps_base 2 --profile_reps 5 \
  --out latency/results/qwen3.5-4b_verih.json

# all six (model, task) pairs
bash latency/run_all_latency.sh 6

# probe-only scaling curves
bash latency/run_scaling.sh 6

# rebuild the CSV/Markdown tables
python latency/merge_latency.py
```

Runtime: ~6–7 min per (model, task) on a shared A6000 (≈40 min for all six),
plus ~30 s per scaling run. Artifacts:
`latency/results/<model>_<task>.json`,
`latency/results/scaling_<model>_<task>.json`,
`latency/results/latency_summary.csv`,
`latency/results/LATENCY_TABLE.md`.

---

## Appendix A. Probe-only scaling runs (device time, ms)

Profiled with `latency/profile_scaling.py` (6 length points, 5 interleaved
trials, minimum per method; probe-only, no base model). This is the cleanest
view of how each probe scales with `n`; the `sp{k}_inc` series is flat, the
standalone series grow with `n`, and MultiMax grows slowly.

**Qwen3.5-4B / VerIH**

| n | MLP | MultiMax | SP-1 | SP-5 | SP-1 KV | SP-5 KV |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.029 | 0.045 | 0.69 | 3.11 | 0.61 | 2.97 |
| 64 | 0.030 | 0.046 | 0.72 | 3.15 | 0.61 | 2.98 |
| 100 | 0.030 | 0.054 | 0.77 | 3.23 | 0.61 | 2.99 |
| 183 | 0.029 | 0.066 | 1.01 | 3.48 | 0.61 | 3.01 |
| 324 | 0.031 | 0.062 | 1.59 | 4.06 | 0.62 | 2.99 |
| 653 | 0.029 | 0.082 | 2.90 | 7.80 | 0.63 | 3.03 |

**Qwen3.5-9B / VerIH**

| n | MLP | MultiMax | SP-1 | SP-5 | SP-1 KV | SP-5 KV |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.038 | 0.047 | 1.09 | 4.84 | 0.95 | 7.03 |
| 64 | 0.039 | 0.049 | 1.10 | 4.89 | 0.94 | 4.64 |
| 100 | 0.040 | 0.058 | 1.19 | 4.97 | 0.94 | 7.02 |
| 183 | 0.040 | 0.077 | 1.51 | 5.32 | 0.95 | 4.67 |
| 324 | 0.040 | 0.074 | 2.76 | 8.85 | 0.95 | 4.69 |
| 653 | 0.040 | 0.103 | 9.77 | 11.03 | 0.96 | 4.72 |

(The `SP-5 KV` column is the profiler number and inherits its multi-step
inflation — alternating ~4.7 and ~7.0 ms; the measured latency, §2, is the
reliable value.)

MiniCPM's probe-only scaling runs were noticeably less stable under co-tenant
contention (e.g. a shorter example measured slower than a longer one in some
points), so we do not reproduce that table here; the robust tercile table in
§6 is the reference for MiniCPM. The qualitative picture is the same: the
standalone probe grows with `n` (1.5 ms at ~50 tokens to 14 ms at ~500), the
`sp{k}_core` variant is ~1.1–1.6 ms at ≤150 tokens, and SP-1 KV is flat at
1.02–1.08 ms over the whole range.

## Appendix B. Artifacts

| path | content |
|---|---|
| `latency/bench_latency.py` | per-example latency + device profiling + calibration |
| `latency/common.py` | model registry, cache/split helpers, timing primitives, profiler |
| `latency/profile_scaling.py` | probe-only length-scaling runs |
| `latency/repair_profile.py` | re-profile the three device-time buckets of a result JSON |
| `latency/merge_latency.py` | build CSV + Markdown tables |
| `latency/results/<model>_<task>.json` | raw per-example samples, buckets, calibrations |
| `latency/results/scaling_<model>_<task>.json` | probe-only scaling |
| `latency/results/latency_summary.csv` | one row per (model, task, method) |
| `latency/results/LATENCY_TABLE.md` | all generated tables |
| `latency/results/logs/` | run logs |
