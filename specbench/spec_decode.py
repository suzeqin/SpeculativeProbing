#!/usr/bin/env python3
"""Speculative-decoding harness with Speculative-Probing probes attached.

Implements greedy speculative decoding for the two head families used in the
paper and measures, per generated sequence:

  * acceptance rate     accepted drafted tokens / drafted tokens
  * acceptance length   generated tokens / decoding rounds (tau, incl. bonus)
  * end-to-end latency  CUDA-event time for the whole generation
  * throughput          tokens per second
  * probe overhead      probe CUDA time per round and its share of total

Head convention (both families). The draft-head position p consumes
(base hidden at token p, embedding of token p+1) and predicts token p+2 - the
paper's shifted pairing. Each decoding round is:

  1. fold-forward       (only after a full-acceptance round) process the bonus
                        token with the base model to restore the KV invariant
  2. bootstrap          one single-token draft-head step at position m-2 with
                        (base hidden at m-2, embedding of the last committed
                        token) -> logits -> first draft token d0
  3. probes             SP-k: k soft-prompt draft-head steps on the draft
                        cache; MLP: last-token MLP; MM: MultiMax over all
                        mid-layer positions. Probes never draft.
  4. drafting           gamma-1 further single-token draft steps, each pairing
                        the previous draft output hidden with the embedding of
                        the last drafted token -> d1..d_{gamma-1}
  5. verification       base-model forward over the gamma drafted tokens;
                        greedy accept the longest matching prefix + one bonus
  6. commit             committed cache update; on rejection the base-model
                        cache is restored from the round-start snapshot and
                        the accepted prefix + bonus are re-forwarded (needed
                        because Qwen3.5's linear-attention layers carry a
                        recurrent state that cannot be rolled back by crop).

State invariants at round start (m = committed length):
  * base cache holds positions 0..m-1; base_hiddens[m-1] is the last hidden
  * last_base_logits predicts token m
  * draft cache holds positions 0..m-3 (one behind: the bootstrap fills m-2)
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "fast_softprompt")):
    if p not in sys.path:
        sys.path.insert(0, p)

BF16 = torch.bfloat16
DEVICE = "cuda"


class CudaSpan:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        return False

    def ms(self) -> float:
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


def cache_crop(cache, n_remove: int):
    """Remove the last n_remove entries (transformers 5.x: negative = remove)."""
    if n_remove > 0:
        cache.crop(-n_remove)


# ---------------------------------------------------------------------------
# Base-model cache snapshot / restore (for rejection rollback)
# ---------------------------------------------------------------------------

_LAYER_TENSORS = ("keys", "values")


def snapshot_cache(cache) -> dict:
    """Deep-copy all mutable state of a (possibly hybrid) transformers cache."""
    snap = {"layers": []}
    for layer in cache.layers:
        entry = {"tensors": {}, "dicts": {}, "scalars": {}}
        for name in _LAYER_TENSORS:
            v = getattr(layer, name, None)
            if torch.is_tensor(v):
                entry["tensors"][name] = v.detach().clone()
        for name in ("recurrent_states", "conv_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                entry["dicts"][name] = {k: t.detach().clone() for k, t in d.items()
                                        if torch.is_tensor(t)}
        for name in ("cumulative_length",):
            if hasattr(layer, name):
                entry["scalars"][name] = getattr(layer, name)
        snap["layers"].append(entry)
    return snap


def restore_cache(cache, snap: dict):
    """Restore a cache from snapshot_cache (in-place on the cache object)."""
    for layer, entry in zip(cache.layers, snap["layers"]):
        for name, t in entry["tensors"].items():
            setattr(layer, name, t.clone())
        for name, d in entry["dicts"].items():
            setattr(layer, name, {k: t.clone() for k, t in d.items()})
        for name, v in entry["scalars"].items():
            setattr(layer, name, v)


# ---------------------------------------------------------------------------
# Shared speculative loop
# ---------------------------------------------------------------------------

class SpecEngineBase:
    """Shared greedy speculative-decoding loop (subclass supplies head ops)."""

    collect_mid = False  # set True to accumulate mid-layer hiddens (MM probe)

    def _draft_bootstrap(self, h_prev, emb, pos):
        """One draft-head step; returns (h_out, next-token logits)."""
        raise NotImplementedError

    def _draft_token_from_logits(self, logits) -> torch.Tensor:
        raise NotImplementedError

    def base_logits(self, h):
        """Base-model logits over hidden states h (subclass may rescale)."""
        return self.lm_head(h)

    def reset(self):
        from transformers import DynamicCache
        self.cache = DynamicCache()          # base-model cache
        self.draft_cache = DynamicCache()    # draft-head cache (full attention)
        self.m = 0
        self.base_hiddens = []
        self.last_base_logits = None
        self.mid_hidden = []
        self.committed = []
        self._snap = None

    # -- engine hooks ----------------------------------------------------------
    def _base_cache_arg(self):
        """MiniCPM's remote code requires an explicit Cache instance."""
        return None

    def _bootstrap_input(self, pos: int) -> torch.Tensor:
        """Draft-head hidden input for the bootstrap step at `pos`."""
        return self.base_hiddens[pos]

    def _append_states(self, out, nrows: int):
        """Append the first `nrows` rows of a base forward to the state."""
        for i in range(nrows):
            self.base_hiddens.append(out.last_hidden_state[:, i:i + 1])
        if self.collect_mid:
            self.mid_hidden.extend(list(out.hidden_states[self.mid_layer][0, :nrows]))

    def _forward_base(self, input_ids: torch.Tensor, collect_mid: bool):
        """Base-model forward over new tokens with the resident cache.

        Default: let the model infer positions. Engines whose remote code
        mis-derives the past length (MiniCPM) override this with explicit
        position_ids and a 4D causal mask."""
        return self.model.model(input_ids=input_ids, use_cache=True,
                                past_key_values=self.cache,
                                output_hidden_states=collect_mid)

    # -- prefill -------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: list[int]):
        self.reset()
        x = torch.tensor([ids], device=DEVICE)
        out = self.model.model(input_ids=x, use_cache=True, output_hidden_states=True,
                               past_key_values=self._base_cache_arg())
        self.cache = out.past_key_values
        self.m = len(ids)
        last = out.last_hidden_state                       # [1, n, d]
        self.base_hiddens = [last[:, i:i + 1] for i in range(self.m)]
        self.last_base_logits = self.base_logits(last[:, -1:])
        if self.collect_mid:
            self.mid_hidden = list(out.hidden_states[self.mid_layer][0])
        self._snap = snapshot_cache(self.cache)
        return out

    # -- round ---------------------------------------------------------------
    @torch.no_grad()
    def round_step(self, probe=None):
        t = {}
        gamma = self.gamma
        m = self.m

        # 1. fold-forward if the bonus token has not been processed yet
        if self.cache.get_seq_length() < self.m:
            with CudaSpan() as s:
                tok = torch.tensor([[self.committed[-1]]], device=DEVICE)
                out = self._forward_base(tok, collect_mid=True)
                self._append_states(out, 1)
                self.last_base_logits = self.base_logits(out.last_hidden_state)
            t["fold_ms"] = s.ms()
        else:
            t["fold_ms"] = 0.0

        # 2. bootstrap draft step at position m-2
        emb_last = self.tok_emb(torch.tensor([[self.committed[-1]]], device=DEVICE))
        with CudaSpan() as s:
            h_out, logits = self._draft_bootstrap(self._bootstrap_input(m - 2).to(BF16),
                                                  emb_last, m - 2)
        t["draft_ms"] = s.ms()
        drafted = [self._draft_token_from_logits(logits).view(1)]

        # 3. probes (off the decoding path; never draft). Runs after the
        #    bootstrap so the draft cache covers 0..m-2 exactly as the trained
        #    classifier incremental path expects.
        t["probe_ms"] = 0.0
        verdict = None
        if probe is not None:
            with CudaSpan() as spr:
                verdict = probe(self)
            t["probe_ms"] = spr.ms()

        # 4. drafting steps
        h_prev = h_out
        with CudaSpan() as s:
            for i in range(1, gamma):
                emb = self.tok_emb(drafted[-1].view(1, 1))
                h_prev, logits_i = self._draft_bootstrap(h_prev, emb, m - 1 + (i - 1))
                drafted.append(self._draft_token_from_logits(logits_i).view(1))
        t["draft_ms"] += s.ms()
        draft_ids = torch.stack(drafted).view(1, -1)       # [1, gamma]

        # 5. verification
        with CudaSpan() as sv:
            vout = self._forward_base(draft_ids, collect_mid=True)
            vlogits = self.base_logits(vout.last_hidden_state)          # [1, gamma, V]
            all_pred = torch.cat([self.last_base_logits, vlogits], dim=1)[0]
            accepted = 0
            for i in range(gamma):
                if int(draft_ids[0, i]) == int(all_pred[i].argmax()):
                    accepted += 1
                else:
                    break
            bonus = int(all_pred[accepted].argmax())
            new_tokens = draft_ids[0, :accepted].tolist() + [bonus]
            self.committed.extend(new_tokens)
            self.m += len(new_tokens)
            m_new = self.m
            self.last_base_logits = vlogits[:, accepted:accepted + 1]
            if accepted < gamma:
                # rejection: roll the base cache back to the round-start
                # snapshot and re-forward the *committed* tokens (accepted
                # drafts + the bonus correction), not the rejected drafts
                restore_cache(self.cache, self._snap)
                commit_ids = torch.cat([draft_ids[:, :accepted],
                                        torch.tensor([[bonus]], device=DEVICE)], dim=1)
                rf = self._forward_base(commit_ids, collect_mid=True)
                self._append_states(rf, accepted + 1)
                self.last_base_logits = self.base_logits(rf.last_hidden_state[:, -1:])
            else:
                # full acceptance: the bonus token was never forwarded; its
                # hidden state arrives with the next round's fold-forward
                self._append_states(vout, accepted)
        t["verify_ms"] = sv.ms()

        # 6. commit the round-start snapshot for the next round
        self._snap = snapshot_cache(self.cache)

        # draft cache: keep positions 0..m_new-3
        want = m_new - 2
        have = self.draft_cache.get_seq_length()
        if have > want:
            cache_crop(self.draft_cache, have - want)

        return {
            "generated": len(new_tokens),
            "drafted": gamma,
            "accepted": accepted,
            "verdict": verdict,
            **t,
        }

    def probe_tail_crop(self, k: int):
        """Drop the k soft-prompt entries a probe appended to the draft cache."""
        cache_crop(self.draft_cache, k)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, probe=None):
        """Full generation loop. Returns (tokens, per-round stats)."""
        self.prefill(prompt_ids)
        self.committed = list(prompt_ids)
        rounds = []
        while self.m - len(prompt_ids) < max_new_tokens:
            st = self.round_step(probe=probe)
            rounds.append(st)
            if self.eos_ids and self.committed[-1] in self.eos_ids:
                break
        return self.committed[len(prompt_ids):], rounds


# ---------------------------------------------------------------------------
# Qwen3.5 MTP engine
# ---------------------------------------------------------------------------

class QwenSpecEngine(SpecEngineBase):
    def __init__(self, base_dir: str, mtp_dir: str, draft_gamma: int = 3,
                 collect_mid: bool = False):
        from transformers import AutoModelForCausalLM
        from qwen35_mtp import Qwen3_5MTP

        self.gamma = draft_gamma
        self.collect_mid = collect_mid
        self.model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=BF16, trust_remote_code=True).to(DEVICE).eval()
        self.lm_head = self.model.lm_head
        self.tok_emb = self.model.model.embed_tokens
        self.mtp = Qwen3_5MTP.from_mtp_checkpoint(mtp_dir, base_dir).to(DEVICE, BF16).eval()
        # The extracted MTP checkpoint has no lm_head/embed_tokens (shared with
        # the base model, tie_word_embeddings=True); the module's own copy is
        # randomly initialised and would produce garbage draft logits.
        with torch.no_grad():
            self.mtp.lm_head.weight = self.lm_head.weight
            self.mtp.model.embed_tokens.weight = self.tok_emb.weight
        self.mid_layer = 8 if "4B" in base_dir else 16
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc is not None else None
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None
        self.reset()

    @torch.no_grad()
    def prefill(self, ids):
        out = super().prefill(ids)
        # draft head over positions 0..m-3 with shifted embeddings (invariant)
        n = self.m
        if n - 2 > 0:
            hs = torch.cat(self.base_hiddens[:n - 2], dim=1)            # [1, n-2, d]
            embeds = self.tok_emb(torch.tensor([ids[1:n - 1]], device=DEVICE))
            self.mtp.model(hidden_states=hs.to(BF16), inputs_embeds=embeds.to(BF16),
                           use_cache=True, past_key_value=self.draft_cache)

    def _draft_bootstrap(self, h_prev, emb, pos):
        pos3 = torch.tensor([[[pos]]], device=DEVICE).expand(3, 1, 1)
        cpos = torch.tensor([pos], device=DEVICE)
        out = self.mtp.model(hidden_states=h_prev.to(BF16), inputs_embeds=emb.to(BF16),
                             use_cache=True, past_key_value=self.draft_cache,
                             position_ids=pos3, cache_position=cpos)
        h = out[0]
        return h, self.mtp.lm_head(h)

    def _draft_token_from_logits(self, logits) -> torch.Tensor:
        return logits.argmax(dim=-1).view(1)


# ---------------------------------------------------------------------------
# MiniCPM4.1 Eagle3 engine
# ---------------------------------------------------------------------------

EAGLE3_LAYER_IDS = (2, 16, 29)


class Eagle3SpecEngine(SpecEngineBase):
    """MiniCPM4.1 Eagle3 draft loop using the repo's classifier conventions
    (the conventions the trained probes and the reproduction use): fc-fused
    hidden + raw embedding, cat([hidden_norm, embed]) into the midlayer.
    Recursive steps chain the previous output hidden directly (official EAGLE3
    recipe: no fc at recursion)."""

    def __init__(self, base_dir: str, eagle3_dir: str, draft_gamma: int = 3,
                 collect_mid: bool = False):
        import minicpm_compat  # noqa: F401
        from transformers import AutoModelForCausalLM
        from minicpm_eagle3_wrapper import from_minicpm_eagle3_checkpoint

        self.gamma = draft_gamma
        self.collect_mid = collect_mid
        self.model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=BF16, trust_remote_code=True).to(DEVICE).eval()
        import minicpm_compat
        minicpm_compat.fix_minicpm_rotary(self.model)
        self.lm_head = self.model.lm_head
        self.tok_emb = self.model.model.embed_tokens
        self.e3 = from_minicpm_eagle3_checkpoint(eagle3_dir).to(DEVICE, BF16).eval()
        self.mid_layer = 16
        # MiniCPM's ForCausalLM rescales hidden states before lm_head
        self.logit_scale = float(self.model.config.hidden_size /
                                 getattr(self.model.config, "dim_model_base", 256))
        eos = getattr(self.model, "generation_config", None)
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None
        self.reset()

    def _base_cache_arg(self):
        from transformers import DynamicCache
        return DynamicCache()

    def reset(self):
        super().reset()
        # per-position Eagle3 fusion hiddens (layers 2/16/29), each [1, 1, d]
        self.e3_h = ([], [], [])

    def _append_states(self, out, nrows: int):
        super()._append_states(out, nrows)
        for li, lid in enumerate(EAGLE3_LAYER_IDS):
            for i in range(nrows):
                self.e3_h[li].append(out.hidden_states[lid][:, i:i + 1])

    def base_logits(self, h):
        return self.lm_head(h / self.logit_scale)

    def _forward_base(self, input_ids: torch.Tensor, collect_mid: bool):
        """MiniCPM remote code derives past_key_values_length only for its
        InfLLMv2 cache, so plain DynamicCache decode steps get RoPE position 0.
        Pass explicit positions and a 4D causal mask instead."""
        from transformers import DynamicCache
        past = self.cache.get_seq_length()
        q = input_ids.size(1)
        pos_ids = torch.arange(past, past + q, device=DEVICE).unsqueeze(0)
        total = past + q
        mask = torch.zeros(1, 1, q, total, device=DEVICE, dtype=torch.bfloat16)
        causal = torch.triu(torch.full((q, q), torch.finfo(torch.bfloat16).min,
                                       device=DEVICE), diagonal=1)
        mask[:, :, :, past:] = causal
        return self.model.model(input_ids=input_ids, use_cache=True,
                                past_key_values=self.cache,
                                position_ids=pos_ids, attention_mask=mask,
                                output_hidden_states=collect_mid)

    def _bootstrap_input(self, pos: int) -> torch.Tensor:
        """fc-fused 3-layer base hidden at `pos` (official Eagle3 prefix input)."""
        fused = self.e3.fc(torch.cat([self.e3_h[li][pos] for li in range(3)], dim=-1))
        return fused

    @torch.no_grad()
    def prefill(self, ids):
        out = super().prefill(ids)
        n = self.m
        # per-position Eagle3 fusion hiddens (needed for bootstraps + probes)
        for li, lid in enumerate(EAGLE3_LAYER_IDS):
            hs_lid = out.hidden_states[lid]
            self.e3_h[li].extend([hs_lid[:, i:i + 1] for i in range(n)])
        if n - 2 > 0:
            lows = [out.hidden_states[lid][:, :n - 2] for lid in EAGLE3_LAYER_IDS]
            combined = torch.cat(lows, dim=-1)
            fused = self.e3.fc(combined)
            hs = self.e3.midlayer.hidden_norm(fused)
            embeds = self.tok_emb(torch.tensor([ids[1:n - 1]], device=DEVICE))
            attn_input = torch.cat([hs, embeds.to(BF16)], dim=-1)
            pos_ids = torch.arange(n - 2, device=DEVICE).unsqueeze(0)
            pos_emb = self.e3.rotary_emb(fused, pos_ids)
            attn_out = self.e3.midlayer.self_attn(hidden_states=attn_input,
                                                  position_embeddings=pos_emb,
                                                  past_key_values=self.draft_cache)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h = fused + attn_out * self.e3.residual_scale
            h = h + self.e3.midlayer.mlp(self.e3.midlayer.post_attention_layernorm(h)) \
                * self.e3.residual_scale

    def _draft_bootstrap(self, h_prev, emb, pos):
        """Recursive Eagle3 step: previous output hidden + token embedding
        (official EAGLE3 recursion feeds the draft's own output hidden straight
        into the midlayer; no fc)."""
        hs = self.e3.midlayer.hidden_norm(h_prev.to(BF16))
        attn_input = torch.cat([hs, emb.to(BF16)], dim=-1)
        pos_ids = torch.tensor([[pos]], device=DEVICE)
        pos_emb = self.e3.rotary_emb(hs, pos_ids)
        attn_out = self.e3.midlayer.self_attn(hidden_states=attn_input,
                                              position_embeddings=pos_emb,
                                              past_key_values=self.draft_cache)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = h_prev.to(BF16) + attn_out * self.e3.residual_scale
        h = h + self.e3.midlayer.mlp(self.e3.midlayer.post_attention_layernorm(h)) \
            * self.e3.residual_scale
        return h, self.e3.lm_head(self.e3.norm(h))

    def _draft_token_from_logits(self, logits) -> torch.Tensor:
        return self.e3.d2t[logits.argmax(dim=-1)].view(1)


# ---------------------------------------------------------------------------
# Greedy (non-speculative) reference engine
# ---------------------------------------------------------------------------

class GreedyEngine:
    def __init__(self, base_dir: str):
        import minicpm_compat  # noqa: F401
        from transformers import AutoModelForCausalLM
        self._needs_explicit_cache = "minicpm" in base_dir.lower()
        self.model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=BF16, trust_remote_code=True).to(DEVICE).eval()
        minicpm_compat.fix_minicpm_rotary(self.model)
        self.lm_head = self.model.lm_head
        self.logit_scale = float(self.model.config.hidden_size /
                                 getattr(self.model.config, "dim_model_base", 256))
        eos = getattr(self.model, "generation_config", None)
        eos = eos.eos_token_id if eos is not None else None
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None

    def _decode_step(self, tok: torch.Tensor, cache):
        """One cached decode step with correct positions/mask.

        MiniCPM's remote code only derives the past length for its InfLLMv2
        cache, so plain DynamicCache decode steps get RoPE position 0 - pass
        explicit position_ids and a 4D mask instead."""
        if not self._needs_explicit_cache:
            return self.model.model(input_ids=tok.view(1, 1), use_cache=True,
                                    past_key_values=cache, output_hidden_states=False)
        past = cache.get_seq_length()
        pos_ids = torch.tensor([[past]], device=DEVICE)
        total = past + 1
        mask = torch.zeros(1, 1, 1, total, device=DEVICE, dtype=torch.bfloat16)
        return self.model.model(input_ids=tok.view(1, 1), use_cache=True,
                                past_key_values=cache, position_ids=pos_ids,
                                attention_mask=mask, output_hidden_states=False)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens):
        x = torch.tensor([prompt_ids], device=DEVICE)
        cache = None
        if self._needs_explicit_cache:
            from transformers import DynamicCache
            cache = DynamicCache()
            out = self.model.model(input_ids=x, use_cache=True,
                                   past_key_values=cache, output_hidden_states=False)
        else:
            out = self.model.model(input_ids=x, use_cache=True, output_hidden_states=False)
            cache = out.past_key_values
        logits = self.lm_head(out.last_hidden_state[:, -1:] / self.logit_scale)
        tokens = []
        events = []
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            tok = logits.argmax()
            tokens.append(int(tok))
            ev = CudaSpan()
            with ev:
                out = self._decode_step(tok, cache)
            events.append(ev)
            logits = self.lm_head(out.last_hidden_state / self.logit_scale)
            if self.eos_ids and tokens[-1] in self.eos_ids:
                break
        total_ms = sum(ev.ms() for ev in events)
        return tokens, {"total_gpu_ms": total_ms, "steps": len(tokens),
                        "wall_s": time.perf_counter() - t0}
