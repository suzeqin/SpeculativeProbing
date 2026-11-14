#!/usr/bin/env python3
"""Measure the real inference-time overhead of the probes.

For one (model, task) pair this script measures, per test example and at
batch size 1:

  base_prefill        base LM forward over the prompt (features are free)
  base_prefill_nocache same without retaining hidden states / KV cache
  base_decode         per-token cost of cached autoregressive decoding
  mlp                 last-token MLP probe  (train_mlp.MLPClassifier)
  mm                  MultiMax probe        (train_multimax.MultiMaxClassifier)
  sp{k}               Speculative Probing, full-sequence draft-head pass
                      (the exact module used in training/eval)
  sp{k}_inc           Speculative Probing marginal cost when the draft head's
                      KV cache over the verified prefix is already resident:
                      k single-token draft-head steps + the linear head
  sp{k}_core          (MiniCPM only) sp{k} without the unused draft-vocabulary
                      lm_head + d2t projection

Everything is timed with per-call CUDA events (L2 flushed between calls) and,
on a subset, with caller-observed CPU-synchronised latency.

Example:
  python latency/bench_latency.py --model qwen3.5-4b --task verih \
      --methods mlp,mm,sp1,sp2,sp5 --out latency/results/qwen3.5-4b_verih.json
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402


# ---------------------------------------------------------------------------
# Incremental (KV-cached) Speculative Probing engines
# ---------------------------------------------------------------------------

class QwenIncrementalSP:
    """SP-k marginal cost for the Qwen MTP head with a resident prefix cache.

    The MTP layer in this repo consumes (base last-layer hidden state, shifted
    token embedding).  In speculative decoding the draft head has already run
    over the verified prefix, so its DynamicCache for positions ``0..n-2`` is
    resident.  Speculative probing only needs the ``k`` soft-prompt positions
    ``n-1 .. n+k-2``: position ``n-1`` uses the real last hidden state plus
    ``sp_0``; each further position uses the previous output hidden state plus
    ``sp_i``.  This mirrors ``RecursiveSoftPromptClassifier.forward`` exactly,
    minus the full-sequence pass that the drafter already performed.
    """

    def __init__(self, clf, device: torch.device):
        self.clf = clf
        self.device = device
        self.mtp = clf.mtp_model
        self.k = int(clf.num_recursive_steps)
        self.dtype = next(self.mtp.parameters()).dtype

    def prepare(self, hidden: torch.Tensor, ids: torch.Tensor) -> Dict[str, Any]:
        from transformers import DynamicCache

        n = int(ids.shape[1])
        embeds = self.clf.embed_weights[ids.to(self.device)]
        cache = DynamicCache()
        with torch.no_grad():
            self.mtp.model(
                hidden_states=hidden[:, : n - 1].to(self.dtype),
                inputs_embeds=embeds[:, 1:n].to(self.dtype),
                use_cache=True,
                past_key_value=cache,
            )
        assert cache.get_seq_length() == n - 1, cache.get_seq_length()
        # Position tensors are loop-invariant metadata; precompute them outside
        # the timed region (a pageable H2D copy costs milliseconds on a busy
        # GPU and is not part of the probe's compute).
        pos3 = []
        cpos = []
        for step in range(self.k):
            pos = n - 1 + step
            pos3.append(torch.tensor([[[pos]]], device=self.device).expand(3, 1, 1))
            cpos.append(torch.tensor([pos], device=self.device))
        return {
            "h": hidden[:, n - 1 : n].to(self.dtype).clone(),
            "cache": cache,
            "n": n,
            "prefix_len": n - 1,
            "pos3": pos3,
            "cpos": cpos,
        }

    def run(self, prep: Dict[str, Any]):
        dev = self.device
        h = prep["h"]
        cache = prep["cache"]
        n = prep["n"]
        for step in range(self.k):
            sp = self.clf.soft_prompts[step].to(dtype=self.dtype)
            out = self.mtp.model(
                hidden_states=h,
                inputs_embeds=sp,
                use_cache=True,
                past_key_value=cache,
                position_ids=prep["pos3"][step],
                cache_position=prep["cpos"][step],
            )
            h = out[0]
        z = self.clf.head(h[:, 0, :].float()).squeeze(-1)
        return torch.stack([-z, z], dim=-1)

    def trim(self, prep: Dict[str, Any]) -> None:
        prep["cache"].crop(-self.k)


class Eagle3IncrementalSP:
    """SP-k marginal cost for the MiniCPM Eagle3 head with a resident cache.

    Same idea as ``QwenIncrementalSP``.  ``Eagle3SoftPromptClassifier`` appends
    ``k`` learned soft prompts (one embedding + three hidden streams each) to
    the sequence and runs the single-layer Eagle3 head over it.  With the
    verified prefix's KV cache resident, only the ``k`` soft-prompt positions
    are evaluated.
    """

    def __init__(self, clf, device: torch.device):
        self.clf = clf
        self.device = device
        self.eagle3 = clf.eagle3
        self.k = int(clf.soft_prompt_embed.size(1))
        self.dtype = next(self.eagle3.parameters()).dtype
        self.sp_h = [clf.soft_prompt_h_low, clf.soft_prompt_h_mid, clf.soft_prompt_h_high]

    def prepare(self, hidden_list: List[torch.Tensor], ids: torch.Tensor) -> Dict[str, Any]:
        from transformers import DynamicCache

        n = int(ids.shape[1])
        embeds = self.clf.embed_weights[ids.to(self.device)]
        cache = DynamicCache()
        with torch.no_grad():
            self.eagle3(
                hidden_states_list=[h[:, : n - 1].to(self.dtype) for h in hidden_list],
                inputs_embeds=embeds[:, 1:n].to(self.dtype),
                use_cache=True,
                past_key_value=cache,
                skip_lm_head=True,
            )
        assert cache.get_seq_length() == n - 1, cache.get_seq_length()
        pos = [torch.tensor([[n - 1 + step]], device=self.device) for step in range(self.k)]
        return {"cache": cache, "n": n, "prefix_len": n - 1, "pos": pos}

    def run(self, prep: Dict[str, Any]):
        cache = prep["cache"]
        n = prep["n"]
        h_out = None
        for step in range(self.k):
            hidden_list = [
                sp[0, step : step + 1, :].to(dtype=self.dtype).view(1, 1, -1)
                for sp in self.sp_h
            ]
            emb = (
                self.clf.soft_prompt_embed[0, step : step + 1, :]
                .to(dtype=self.dtype)
                .view(1, 1, -1)
            )
            h_out = self.eagle3(
                hidden_states_list=hidden_list,
                inputs_embeds=emb,
                use_cache=True,
                past_key_value=cache,
                position_ids=prep["pos"][step],
                skip_lm_head=True,
            ).hidden_states
        z = self.clf.head(h_out[:, 0, :].float()).squeeze(-1)
        return torch.stack([-z, z], dim=-1)

    def trim(self, prep: Dict[str, Any]) -> None:
        prep["cache"].crop(-self.k)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_task_data(spec: Dict[str, Any], task: str, device: torch.device,
                      limit: Optional[int] = None, stride: int = 1,
                      indices: Optional[List[int]] = None) -> Dict[str, Any]:
    caches = spec["caches"]
    meta_last = C.load_cache_meta(C.abs_path(caches["last"].format(task=task)))
    meta_mid = C.load_cache_meta(C.abs_path(caches["mid"].format(task=task)))
    num_train = meta_last["num_train"]
    num_test = meta_last["num_test"]
    idxs = C.split_test_indices(num_train, num_test, C.TASK_TARGET_EVAL[task])
    n_eval = num_test + max(0, C.TASK_TARGET_EVAL[task] - num_test)
    assert len(idxs) == n_eval - int(round(0.5 * n_eval)), (len(idxs), n_eval)
    if indices is not None:
        idxs = list(indices)
    else:
        if stride > 1:
            idxs = idxs[::stride]
        if limit is not None:
            idxs = idxs[:limit]

    ids_cpu = meta_last["token_ids"]
    hidden_last = []
    hidden_mid = []
    token_ids = []
    masks = []
    for i in idxs:
        h = meta_last["hidden_states"][i].to(device=device, dtype=torch.bfloat16)
        hidden_last.append(h.unsqueeze(0))
        m = meta_mid["hidden_states"][i].to(device=device, dtype=torch.bfloat16)
        hidden_mid.append(m.unsqueeze(0))
        t = torch.tensor(ids_cpu[i], dtype=torch.long, device=device).unsqueeze(0)
        token_ids.append(t)
        masks.append(torch.ones_like(t))

    e3_hidden = None
    if "eagle3" in caches:
        meta_e3 = C.load_cache_meta(C.abs_path(caches["eagle3"].format(task=task)))
        n_layers = int(meta_e3.get("eagle3_num_layers", 3))
        d_layer = int(meta_e3.get("eagle3_per_layer_dim", meta_e3["hidden_dim"] // n_layers))
        e3_hidden = []
        for i in idxs:
            flat = meta_e3["hidden_states"][i]  # [n, 3d] mmap view
            e3_hidden.append(
                [
                    flat[:, li * d_layer : (li + 1) * d_layer]
                    .to(device=device, dtype=torch.bfloat16)
                    .unsqueeze(0)
                    for li in range(n_layers)
                ]
            )
        meta_eagle3 = meta_e3
    else:
        meta_eagle3 = None

    lengths = [int(t.shape[1]) for t in token_ids]
    return {
        "idxs": idxs,
        "lengths": lengths,
        "hidden_last": hidden_last,
        "hidden_mid": hidden_mid,
        "eagle3_hidden": e3_hidden,
        "token_ids": token_ids,
        "masks": masks,
        "meta_last": meta_last,
        "meta_mid": meta_mid,
        "meta_eagle3": meta_eagle3,
        "num_train": num_train,
        "num_test": num_test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_methods(spec: Dict[str, Any], data: Dict[str, Any], device: torch.device,
                  method_names: List[str], seed: int):
    """Build every requested probe.

    Returns (methods, prepare, release, between, keepalive):
      methods   name -> fn(i)   timed call
      prepare   name -> fn(i)   (optional) build per-example state outside timing
      release   name -> fn(i)   (optional) free per-example state
      between   name -> fn()    (optional) reset state between timed repetitions
    """
    methods: Dict[str, Callable[[int], Any]] = {}
    prepare: Dict[str, Callable[[int], None]] = {}
    release: Dict[str, Callable[[int], None]] = {}
    between: Dict[str, Callable[[], None]] = {}
    keepalive: Dict[str, Any] = {}
    n_ex = len(data["lengths"])

    if any(m in method_names for m in ("mlp", "mm")):
        d_last = data["hidden_last"][0].shape[-1]
        d_mid = data["hidden_mid"][0].shape[-1]
        if "mlp" in method_names:
            probe = C.build_mlp(d_last, device, seed=seed)
            last_vec = [h[:, -1, :] for h in data["hidden_last"]]
            methods["mlp"] = lambda i, p=probe, v=last_vec: p(v[i])
            keepalive["mlp"] = probe
            keepalive.setdefault("params", {})["mlp"] = sum(x.numel() for x in probe.parameters())
        if "mm" in method_names:
            probe = C.build_multimax(d_mid, device, seed=seed)
            hmid = data["hidden_mid"]
            masks = data["masks"]
            methods["mm"] = lambda i, p=probe, h=hmid, m=masks: p(h[i], m[i])
            keepalive["mm"] = probe
            keepalive.setdefault("params", {})["mm"] = sum(x.numel() for x in probe.parameters())

    sp_ks = sorted(
        int(m[2]) for m in method_names
        if m.startswith("sp") and not m.endswith("_inc") and not m.endswith("_core")
    )
    core_ks = sorted(int(m[2]) for m in method_names if m.endswith("_core"))
    inc_ks = sorted(int(m[2]) for m in method_names if m.endswith("_inc"))
    if not (sp_ks or inc_ks or core_ks):
        return methods, prepare, release, between, keepalive

    bundle = C.build_sp_bundle(spec, device)
    keepalive["bundle"] = bundle

    if spec["head_kind"] == "mtp":
        hlast = data["hidden_last"]
        ids = data["token_ids"]
        masks = data["masks"]
        for k in sorted(set(sp_ks) | set(inc_ks)):
            clf = bundle.build(k, seed=seed)
            keepalive[f"clf{k}"] = clf
            keepalive.setdefault("params", {})[f"sp{k}"] = sum(
                x.numel() for x in clf.parameters() if x.requires_grad)
            if k in sp_ks:
                methods[f"sp{k}"] = lambda i, c=clf, h=hlast, t=ids, m=masks: c(h[i], t[i], m[i], None)
            if k in inc_ks:
                eng = QwenIncrementalSP(clf, device)
                keepalive[f"eng{k}"] = eng
                preps: List[Any] = [None] * n_ex

                def prep(i, eng=eng, preps=preps):
                    if preps[i] is None:
                        preps[i] = eng.prepare(hlast[i], ids[i])

                def run(i, eng=eng, preps=preps):
                    return eng.run(preps[i])

                def rel(i, preps=preps):
                    preps[i] = None

                name = f"sp{k}_inc"
                # trims the cache of the *current* example; the loop updates the index
                state = {"i": 0}
                between[name] = lambda eng=eng, preps=preps, st=state: (
                    eng.trim(preps[st["i"]]) if preps[st["i"]] is not None else None
                )
                methods[name] = run
                keepalive[f"between_state{k}"] = state
                # expose the per-example index updater through prepare
                def prep_with_index(i, prep=prep, st=state):
                    st["i"] = i
                    prep(i)
                prepare[name] = prep_with_index
                release[name] = rel
    else:
        e3 = data["eagle3_hidden"]
        ids = data["token_ids"]
        masks = data["masks"]
        for k in sorted(set(sp_ks) | set(core_ks) | set(inc_ks)):
            clf = bundle.build(k, seed=seed)
            keepalive[f"clf{k}"] = clf
            keepalive.setdefault("params", {})[f"sp{k}"] = sum(
                x.numel() for x in clf.parameters() if x.requires_grad)
            if k in sp_ks:
                methods[f"sp{k}"] = lambda i, c=clf, e=e3, t=ids, m=masks: c(e[i], t[i], m[i], None)
            if k in core_ks:
                methods[f"sp{k}_core"] = lambda i, c=clf, e=e3, t=ids, m=masks: c(
                    e[i], t[i], m[i], None, skip_lm_head=True
                )
            if k in inc_ks:
                eng = Eagle3IncrementalSP(clf, device)
                keepalive[f"eng{k}"] = eng
                preps: List[Any] = [None] * n_ex

                def prep(i, eng=eng, preps=preps):
                    if preps[i] is None:
                        preps[i] = eng.prepare(e3[i], ids[i])

                def run(i, eng=eng, preps=preps):
                    return eng.run(preps[i])

                def rel(i, preps=preps):
                    preps[i] = None

                name = f"sp{k}_inc"
                state = {"i": 0}
                between[name] = lambda eng=eng, preps=preps, st=state: (
                    eng.trim(preps[st["i"]]) if preps[st["i"]] is not None else None
                )
                methods[name] = run
                release[name] = rel
                keepalive[f"between_state{k}"] = state

                def prep_with_index(i, prep=prep, st=state):
                    st["i"] = i
                    prep(i)
                prepare[name] = prep_with_index
    return methods, prepare, release, between, keepalive


def base_forward(spec, base_model, input_ids, attention_mask=None,
                 output_hidden_states=True, use_cache=True, past_key_values=None):
    """Text-backbone forward that works for both model families.

    MiniCPM's remote code requires an explicit ``Cache`` instance whenever
    ``use_cache=True`` (the reproduction's precompute used ``use_cache=False``,
    which is why this was never hit before).
    """
    kw: Dict[str, Any] = {
        "input_ids": input_ids,
        "output_hidden_states": output_hidden_states,
        "use_cache": use_cache,
    }
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    if past_key_values is not None:
        kw["past_key_values"] = past_key_values
    elif use_cache and spec["head_kind"] == "eagle3":
        from transformers import DynamicCache
        kw["past_key_values"] = DynamicCache()
    return base_model.model(**kw)


def load_base_model(spec: Dict[str, Any], device: torch.device):
    import minicpm_compat  # noqa: F401
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        C.abs_path(spec["base_dir"]),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=str(device),
    )
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(C.MODEL_SPECS))
    ap.add_argument("--task", required=True, choices=sorted(C.TASK_TARGET_EVAL))
    ap.add_argument("--methods", default="mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps_base", type=int, default=2)
    ap.add_argument("--reps_cpu", type=int, default=3)
    ap.add_argument("--cpu_examples", type=int, default=6)
    ap.add_argument("--decode_examples", type=int, default=10)
    ap.add_argument("--profile_reps", type=int, default=5)
    ap.add_argument("--no_profile", action="store_true")
    ap.add_argument("--decode_tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--no_flush", action="store_true")
    ap.add_argument("--no_calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    spec = C.MODEL_SPECS[args.model]
    methods_wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_path = args.out or os.path.join(
        HERE, "results", f"{args.model}_{args.task}{args.tag}.json"
    )

    print(f"[{args.model}/{args.task}] device={device} "
          f"({torch.cuda.get_device_name(device)})")
    timer = C.Timer(device, flush_l2=not args.no_flush)

    result: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "model_display": spec["display"],
            "task": args.task,
            "methods": methods_wanted,
            "reps": args.reps,
            "warmup": args.warmup,
            "reps_base": args.reps_base,
            "reps_cpu": args.reps_cpu,
            "decode_tokens": args.decode_tokens,
            "limit": args.limit,
            "stride": args.stride,
            "seed": args.seed,
            "flush_l2": not args.no_flush,
            "with_base": not args.no_base,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "gpu_start": C.gpu_snapshot(torch.cuda.current_device()),
        "status": "started",
    }
    if not args.no_calibrate:
        print("  calibrating reference kernels...")
        result["calibration_start"] = C.calibrate(device, timer)
        print("   ", {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in result["calibration_start"].items() if k != "raw"})

    t_start = time.time()
    try:
        print(f"  loading caches for {args.task}...")
        data = prepare_task_data(spec, args.task, device, limit=args.limit, stride=args.stride)
        n_ex = len(data["lengths"])
        print(f"  {n_ex} test examples, token length mean={sum(data['lengths'])/n_ex:.1f} "
              f"min={min(data['lengths'])} max={max(data['lengths'])}")
        result["num_examples"] = n_ex
        result["lengths"] = data["lengths"]
        result["num_train"] = data["num_train"]
        result["num_test"] = data["num_test"]
        result["accuracy_reproduced"] = C.load_reproduction_accuracy(args.task, spec["result_tag"])

        print("  building probes...")
        methods, prepare_hooks, release_hooks, between, keepalive = build_methods(
            spec, data, device, methods_wanted, args.seed)
        method_names = [m for m in methods_wanted if m in methods]
        missing = [m for m in methods_wanted if m not in methods]
        if missing:
            print(f"  WARNING: could not build {missing}")
        result["methods_requested"] = args.methods
        result["methods_measured"] = method_names

        result["probe_params"] = keepalive.get("params", {})
        base_model = None
        if not args.no_base:
            print("  loading base model...")
            base_model = load_base_model(spec, device)
            print(f"  base model loaded, GPU alloc={torch.cuda.memory_allocated()/1e9:.2f} GB")

        # warm up all methods once on the longest example
        longest = max(range(n_ex), key=lambda i: data["lengths"][i])
        print(f"  warmup on example {longest} (len={data['lengths'][longest]})...")
        if base_model is not None:
            with torch.no_grad():
                _ = base_forward(spec, base_model, data["token_ids"][longest],
                                 data["masks"][longest], output_hidden_states=True, use_cache=True)
        for name in method_names:
            try:
                if name in prepare_hooks:
                    prepare_hooks[name](longest)
                with torch.no_grad():
                    methods[name](longest)
                if name in between:
                    between[name]()
                if name in release_hooks:
                    release_hooks[name](longest)
            except Exception:
                traceback.print_exc()
                raise
        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # per-example timing loop (base and probes interleaved)
        # ------------------------------------------------------------------
        per_ex: Dict[str, List[float]] = {m: [] for m in method_names}
        raw: Dict[str, List[float]] = {m: [] for m in method_names}
        per_ex_base: Dict[str, List[float]] = {}
        raw_base: List[float] = []
        if base_model is not None:
            per_ex_base["prefill"] = []

        cpu_subset = sorted(set(
            [min(n_ex - 1, int((j + 0.5) * n_ex / max(1, args.cpu_examples))) for j in range(args.cpu_examples)]
        ))
        cpu_times: Dict[str, List[List[float]]] = {m: [] for m in method_names}
        cpu_base: Dict[str, List[float]] = {}

        t_loop = time.time()
        for j in range(n_ex):
            with torch.no_grad():
                pending: List[Any] = []  # (name, start_event, end_event)
                if base_model is not None:
                    ids_j, mask_j = data["token_ids"][j], data["masks"][j]

                    def prefill_fn(ids_j=ids_j, mask_j=mask_j):
                        return base_forward(spec, base_model, ids_j, mask_j,
                                            output_hidden_states=True, use_cache=True)

                    for _ in range(args.reps_base):
                        timer._flush()
                        ev_s = torch.cuda.Event(enable_timing=True)
                        ev_e = torch.cuda.Event(enable_timing=True)
                        ev_s.record()
                        prefill_fn()
                        ev_e.record()
                        pending.append(("base_prefill", ev_s, ev_e))

                for name in method_names:
                    fn = methods[name]
                    if name in prepare_hooks:
                        prepare_hooks[name](j)
                    fn(j)  # untimed warmup for this shape
                    for _ in range(args.reps):
                        if name in between:
                            between[name]()
                        timer._flush()
                        ev_s = torch.cuda.Event(enable_timing=True)
                        ev_e = torch.cuda.Event(enable_timing=True)
                        ev_s.record()
                        fn(j)
                        ev_e.record()
                        pending.append((name, ev_s, ev_e))
                    if name in between:
                        between[name]()  # ensure clean state

                if j in cpu_subset:
                    for name in method_names:
                        fn = methods[name]
                        if name in prepare_hooks:
                            prepare_hooks[name](j)
                        cpu_times[name].append(
                            timer.time_cpu(lambda fn=fn, j=j: fn(j), reps=args.reps_cpu, warmup=0))
                        if name in release_hooks:
                            release_hooks[name](j)
                    if base_model is not None:
                        ids_j, mask_j = data["token_ids"][j], data["masks"][j]
                        cpu_base.setdefault("prefill", []).append(
                            C.summarize(timer.time_cpu(
                                lambda ids_j=ids_j, mask_j=mask_j: base_forward(
                                    spec, base_model, ids_j, mask_j,
                                    output_hidden_states=True, use_cache=True),
                                reps=max(2, args.reps_cpu // 2), warmup=0))["median"]
                        )

                torch.cuda.synchronize()
                by_name: Dict[str, List[float]] = {}
                for name, ev_s, ev_e in pending:
                    by_name.setdefault(name, []).append(ev_s.elapsed_time(ev_e) * 1e3)
                for name in method_names:
                    ts = by_name.get(name, [])
                    if ts:
                        # Per-example estimate = fastest of ``reps`` samples, so
                        # that bursts of contention from co-tenants do not
                        # dominate the aggregate (see report notes).
                        per_ex[name].append(min(ts))
                        raw[name].extend(round(x, 3) for x in ts)
                    else:
                        per_ex[name].append(float("nan"))
                if base_model is not None:
                    ts = by_name.get("base_prefill", [])
                    if ts:
                        per_ex_base["prefill"].append(min(ts))
                        raw_base.extend(round(x, 3) for x in ts)
                    else:
                        per_ex_base["prefill"].append(float("nan"))

                for name in method_names:
                    if name in release_hooks:
                        release_hooks[name](j)
            if (j + 1) % 50 == 0:
                print(f"    {j+1}/{n_ex} examples ({time.time()-t_loop:.1f}s)",
                      flush=True)
        torch.cuda.synchronize()
        result["loop_seconds"] = time.time() - t_loop

        result["per_example_us"] = per_ex
        result["raw_samples_us"] = raw
        result["per_example_base_us"] = per_ex_base
        result["summary_us"] = {
            # all raw samples pooled (each is one CUDA-event window)
            "probes": {m: C.summarize(v) for m, v in raw.items()},
            "base": ({"prefill": C.summarize(raw_base)} if raw_base
                     else {m: C.summarize(v) for m, v in per_ex_base.items()}),
            "per_example_min": {m: C.summarize(v) for m, v in per_ex.items()},
            "base_per_example_min": {m: C.summarize(v) for m, v in per_ex_base.items()},
        }
        result["cpu_sync_us"] = {
            m: C.summarize([x for run in runs for x in run]) if runs else {}
            for m, runs in cpu_times.items()
        }
        if cpu_base:
            result["cpu_sync_us"]["base_prefill"] = C.summarize(
                [x for run in cpu_base.values() for x in run])

        # ------------------------------------------------------------------
        # peak memory of a single probe call on the longest example
        # ------------------------------------------------------------------
        mem = {}
        for name in method_names:
            if name in prepare_hooks:
                prepare_hooks[name](longest)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            with torch.no_grad():
                methods[name](longest)
            torch.cuda.synchronize()
            mem[name] = round((torch.cuda.max_memory_allocated() - base_alloc) / 1e6, 2)
            if name in between:
                between[name]()
            if name in release_hooks:
                release_hooks[name](longest)
        result["probe_peak_mem_mb"] = mem

        # ------------------------------------------------------------------
        # base decode cost (cached, 1 token per step)
        # ------------------------------------------------------------------
        if base_model is not None:
            step = max(1, n_ex // max(1, args.decode_examples))
            decode_idxs = list(range(0, n_ex, step))[: args.decode_examples]
            per_tok: List[float] = []
            dtypes = []
            for j in decode_idxs:
                ids_j = data["token_ids"][j]
                mask_j = data["masks"][j]
                n = int(ids_j.shape[1])
                with torch.no_grad():
                    out = base_forward(spec, base_model, ids_j, mask_j,
                                       output_hidden_states=False, use_cache=True)
                    cache = out.past_key_values
                    next_id = torch.zeros((1, 1), dtype=torch.long, device=device)
                    ext_mask = torch.ones((1, n + args.decode_tokens), dtype=torch.long, device=device)

                    def decode_fn(cache=cache, next_id=next_id, n=n):
                        # No padding: the cache holds the whole (single) prefix,
                        # so no attention mask is needed and cache_position is
                        # inferred from the cache length.
                        return base_forward(spec, base_model, next_id, None,
                                            output_hidden_states=False, use_cache=True,
                                            past_key_values=cache)

                    ts = timer.time_gpu(decode_fn, reps=args.decode_tokens, warmup=2)
                per_tok.append(sum(ts) / len(ts))
                if (not args.no_profile) and "base_decode" not in result.get("device_time_us", {}) and not args.no_profile:
                    with torch.no_grad():
                        dprof = C.profile_calls(decode_fn, trials=3, warmup=2)
                    result.setdefault("device_time_us", {})["base_decode"] = dprof
                del cache, out
                torch.cuda.empty_cache()
            result["decode_us_per_token"] = C.summarize(per_tok)
            result["decode_examples"] = [data["lengths"][j] for j in decode_idxs]

        # ------------------------------------------------------------------
        # device-side (CUDA kernel) time per call, via torch.profiler
        # ------------------------------------------------------------------
        if not args.no_profile:
            lengths_sorted = sorted(data["lengths"])
            n_s = len(lengths_sorted)
            targets = {
                "short": lengths_sorted[int(0.10 * (n_s - 1))],
                "median": lengths_sorted[n_s // 2],
                "long": lengths_sorted[int(0.90 * (n_s - 1))],
            }
            by_len: Dict[str, Any] = {}
            for bucket, target in targets.items():
                idx_b = min(range(n_ex), key=lambda i: abs(data["lengths"][i] - target))
                entry: Dict[str, Any] = {"index": idx_b, "length": data["lengths"][idx_b], "methods": {}}
                # Prepare KV-cache preps once for this example.
                for name in method_names:
                    if name in prepare_hooks:
                        prepare_hooks[name](idx_b)
                ids_b, mask_b = data["token_ids"][idx_b], data["masks"][idx_b]

                def prefill_prof(ids_b=ids_b, mask_b=mask_b):
                    return base_forward(spec, base_model, ids_b, mask_b,
                                        output_hidden_states=True, use_cache=True)

                def prefill_nocache_prof(ids_b=ids_b, mask_b=mask_b):
                    return base_forward(spec, base_model, ids_b, mask_b,
                                        output_hidden_states=False, use_cache=False)

                samples: Dict[str, List[float]] = {name: [] for name in method_names}
                if base_model is not None:
                    samples["base_prefill"] = []
                    samples["base_prefill_nocache"] = []
                # Untimed warmup, then restore any per-example state (the
                # incremental engines extend their KV cache by k on each run;
                # trim it back before the first profiled call).
                for name in method_names:
                    methods[name](idx_b)
                    if name in between:
                        between[name]()
                # Interleave methods across trials so that a co-tenant's
                # contention burst does not bias one method relative to another.
                for _t in range(args.profile_reps):
                    for name in method_names:
                        samples[name].append(C.profile_one_call(
                            lambda fn=methods[name], i=idx_b: fn(i))["cuda_us"])
                        if name in between:
                            between[name]()
                    if base_model is not None:
                        samples["base_prefill"].append(C.profile_one_call(prefill_prof)["cuda_us"])
                        samples["base_prefill_nocache"].append(
                            C.profile_one_call(prefill_nocache_prof)["cuda_us"])
                nops = {}
                for name, vals in samples.items():
                    entry["methods"][name] = C.summarize_profile_samples(vals)
                by_len[bucket] = entry
                print(f"  device time [{bucket} n={entry['length']}]:",
                      {m: round(v["cuda_us_per_call"], 1) for m, v in entry["methods"].items()},
                      flush=True)
                for name in method_names:
                    if name in release_hooks:
                        release_hooks[name](idx_b)
            result["device_time_us_by_length"] = by_len
            result["profile_example"] = by_len["median"]
            dev_all = result.setdefault("device_time_us", {})
            dev_all.update(by_len["median"]["methods"])

        # ------------------------------------------------------------------
        # wrap-up
        # ------------------------------------------------------------------
        result["elapsed_seconds"] = time.time() - t_start
        result["gpu_end"] = C.gpu_snapshot(torch.cuda.current_device())
        if not args.no_calibrate:
            result["calibration_end"] = C.calibrate(device, timer)
        result["status"] = "ok"

        # derived per-example ratios (probe / base prefill), same examples
        if per_ex_base.get("prefill"):
            base_arr = per_ex_base["prefill"]
            ratios = {}
            for m, arr in per_ex.items():
                ratios[m] = [
                    (arr[i] / base_arr[i]) if base_arr[i] > 0 else float("nan") for i in range(n_ex)
                ]
            result["ratio_to_prefill_per_example"] = ratios
            result["overhead_pct_of_prefill"] = {
                m: 100.0 * sum(arr) / sum(base_arr) for m, arr in per_ex.items()
            }
            result["median_ratio_pct_of_prefill"] = {
                m: 100.0 * C.summarize(ratios[m])["median"] for m in ratios
            }

        C.write_json(out_path, result)
        print(f"\n  wrote {out_path}  ({result['elapsed_seconds']:.1f}s)")
        print("  medians (us):")
        for m in method_names:
            s = result["summary_us"]["probes"][m]
            print(f"    {m:10s} median={s['median']:9.1f}  mean={s['mean']:9.1f}  min={s['min']:9.1f}")
        for m, s in result["summary_us"]["base"].items():
            print(f"    {m:10s} median={s['median']:9.1f}  mean={s['mean']:9.1f}  min={s['min']:9.1f}")
        if "decode_us_per_token" in result:
            print(f"    decode/tok median={result['decode_us_per_token']['median']:.1f}")
        if "overhead_pct_of_prefill" in result:
            print("  overhead % of prefill (sum/sum):",
                  {m: round(v, 3) for m, v in result["overhead_pct_of_prefill"].items()})
        return 0
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        C.write_json(out_path, result)
        return 1


if __name__ == "__main__":
    sys.exit(main())
