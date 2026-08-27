"""Cache I/O helpers.

New (mmap-friendly) format stores hidden_states as a single flat tensor so that
torch.load(..., mmap=True) actually works:

    <path>          pickled dict with everything except hidden_states; includes
                    an `offsets` LongTensor of length (N+1) and
                    `hidden_dim`/`hidden_dtype` metadata.
    <path>.hidden   a single safe-to-mmap file containing the flat
                    [total_tokens, hidden_dim] tensor under key "hidden_flat".

Old (legacy) format stores hidden_states as a Python list of tensors under the
"hidden_states" key. Still readable via load_cache(..., mmap=False).
"""
from __future__ import annotations

import os
from typing import Any, Dict

import torch


class LazyHiddenStates:
    """List-of-tensors view over a flat [total_tokens, hidden_dim] tensor.

    `__getitem__` returns a tensor view (zero-copy) into the mmap'd flat
    storage, so iterating over many samples does not materialize them all.
    """

    def __init__(self, flat: torch.Tensor, offsets: torch.Tensor):
        if flat.ndim != 2:
            raise ValueError(f"flat must be 2D, got {flat.shape}")
        if offsets.ndim != 1:
            raise ValueError(f"offsets must be 1D, got {offsets.shape}")
        self._flat = flat
        self._offsets = offsets
        self._n = int(offsets.numel()) - 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._n)
            return [self[i] for i in range(start, stop, step)]
        if idx < 0:
            idx += self._n
        s = int(self._offsets[idx].item())
        e = int(self._offsets[idx + 1].item())
        return self._flat[s:e]


def save_cache_mmap(path: str, cache: Dict[str, Any]) -> None:
    """Write cache in the new mmap-friendly format.

    Required keys in `cache`: hidden_states (list[Tensor[seq_i, d]]),
    plus whatever task-specific metadata (token_ids, labels, num_train, ...).
    """
    hidden_states = cache["hidden_states"]
    if not hidden_states:
        raise ValueError("hidden_states is empty")
    hidden_dim = hidden_states[0].shape[-1]
    dtype = hidden_states[0].dtype

    # Build offsets and flat tensor.
    lens = [int(h.shape[0]) for h in hidden_states]
    total = sum(lens)
    offsets = torch.empty(len(lens) + 1, dtype=torch.long)
    offsets[0] = 0
    acc = 0
    for i, L in enumerate(lens):
        acc += L
        offsets[i + 1] = acc

    flat = torch.empty((total, hidden_dim), dtype=dtype)
    pos = 0
    for h in hidden_states:
        L = int(h.shape[0])
        flat[pos:pos + L] = h.to(dtype)
        pos += L

    # Persist the flat tensor under a state-dict so torch.load(mmap=True) covers it.
    hidden_path = path + ".hidden"
    torch.save({"hidden_flat": flat}, hidden_path)

    # Write meta (everything else + offsets). Hidden state is now external.
    meta = {k: v for k, v in cache.items() if k != "hidden_states"}
    meta["cache_format"] = "mmap_v1"
    meta["hidden_dim"] = hidden_dim
    meta["hidden_dtype"] = str(dtype).replace("torch.", "")
    meta["offsets"] = offsets
    torch.save(meta, path)


def _resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


class StreamingCacheWriter:
    """Stream per-sample hidden states directly to disk.

    Avoids building a Python list of N tensors in RAM. Instead:
      1. Compute total_tokens by summing caller-supplied seq lengths in advance.
      2. Allocate a flat [total_tokens, hidden_dim] bfloat16 tensor.
      3. Caller appends samples; each write goes into the pre-allocated tensor.
      4. On close(), persist the flat tensor (via torch.save) + meta (offsets,
         task-specific metadata).

    The big flat tensor stays in RAM (if you allocate via torch.empty) or on disk
    (if you allocate via torch.UntypedStorage.from_file(..., shared=True) below).
    The disk-backed variant is used by default so RSS stays low.
    """

    def __init__(self, output_path: str, total_tokens: int, hidden_dim: int,
                 num_samples: int, dtype=torch.bfloat16, backing: str = "mem"):
        self.output_path = output_path
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.total_tokens = total_tokens
        self.dtype = dtype
        self.offsets = torch.empty(num_samples + 1, dtype=torch.long)
        self.offsets[0] = 0
        self._pos = 0
        self._n = 0
        self._meta = None
        self._backing = backing

        if backing == "disk":
            # File-backed storage. Writes go through the kernel page cache and
            # are asynchronously flushed; RSS grows only with the touched pages.
            hidden_path = output_path + ".hidden.raw"
            self._raw_hidden_path = hidden_path
            os.makedirs(os.path.dirname(os.path.abspath(hidden_path)) or ".", exist_ok=True)
            elem_size = torch.tensor([], dtype=dtype).element_size()
            n_bytes = total_tokens * hidden_dim * elem_size
            storage = torch.UntypedStorage.from_file(hidden_path, shared=True, nbytes=n_bytes)
            self.flat = torch.empty(0, dtype=dtype).set_(
                storage, storage_offset=0,
                size=(total_tokens, hidden_dim),
                stride=(hidden_dim, 1),
            )
        else:
            self.flat = torch.empty((total_tokens, hidden_dim), dtype=dtype)

    def append(self, h: torch.Tensor) -> None:
        """Append one sample's hidden states [seq_len, hidden_dim]."""
        if self._n >= self.num_samples:
            raise RuntimeError("Writer received more samples than expected")
        seq_len = int(h.shape[0])
        end = self._pos + seq_len
        self.flat[self._pos:end] = h.to(self.dtype)
        self._pos = end
        self._n += 1
        self.offsets[self._n] = end

    def set_meta(self, meta: Dict[str, Any]) -> None:
        self._meta = dict(meta)

    def close(self) -> None:
        if self._n != self.num_samples:
            raise RuntimeError(f"Expected {self.num_samples} samples, got {self._n}")
        if self._pos != self.total_tokens:
            raise RuntimeError(f"Expected {self.total_tokens} tokens, got {self._pos}")
        # Persist the flat tensor under a state_dict so torch.load(mmap=True) covers it.
        hidden_path = self.output_path + ".hidden"
        torch.save({"hidden_flat": self.flat}, hidden_path)
        if self._backing == "disk" and getattr(self, "_raw_hidden_path", None):
            # Remove the raw backing file; the .hidden torch-save version is canonical.
            try:
                os.remove(self._raw_hidden_path)
            except OSError:
                pass

        meta = dict(self._meta or {})
        meta["cache_format"] = "mmap_v1"
        meta["hidden_dim"] = self.hidden_dim
        meta["hidden_dtype"] = str(self.dtype).replace("torch.", "")
        meta["offsets"] = self.offsets
        torch.save(meta, self.output_path)


class GemmaStreamingCacheWriter:
    """Gemma4-specific variant of the streaming writer.

    Writes five flat tensors (shared per-token offsets) to separate files:
      <path>.hidden       [total_tokens, 2560]   base model's last hidden state
      <path>.full_k       [total_tokens, kv_heads * global_head_dim]   flattened K for full-attn
      <path>.full_v       [total_tokens, kv_heads * global_head_dim]   flattened V for full-attn
      <path>.slid_k       [total_tokens, kv_heads * head_dim]          flattened K for sliding-attn
      <path>.slid_v       [total_tokens, kv_heads * head_dim]          flattened V for sliding-attn

    Metadata stored in <path>:
      cache_format='gemma_mtp_v1', hidden_dim, kv_num_heads,
      full_head_dim, slid_head_dim, offsets, + task-specific keys.
    """

    FIELDS = ("hidden", "full_k", "full_v", "slid_k", "slid_v")

    def __init__(
        self,
        output_path: str,
        total_tokens: int,
        num_samples: int,
        hidden_dim: int,
        kv_num_heads: int,
        full_head_dim: int,
        slid_head_dim: int,
        dtype=torch.bfloat16,
    ):
        self.output_path = output_path
        self.total_tokens = total_tokens
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.kv_num_heads = kv_num_heads
        self.full_head_dim = full_head_dim
        self.slid_head_dim = slid_head_dim
        self.dtype = dtype
        self.offsets = torch.empty(num_samples + 1, dtype=torch.long)
        self.offsets[0] = 0
        self._pos = 0
        self._n = 0
        self._meta = None
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        self._tensors: Dict[str, torch.Tensor] = {}
        self._raw_paths: Dict[str, str] = {}

        def _alloc(field: str, feature_dim: int) -> torch.Tensor:
            raw_path = output_path + f".{field}.raw"
            self._raw_paths[field] = raw_path
            storage = torch.UntypedStorage.from_file(
                raw_path, shared=True, nbytes=total_tokens * feature_dim * elem_size
            )
            return torch.empty(0, dtype=dtype).set_(
                storage, storage_offset=0,
                size=(total_tokens, feature_dim),
                stride=(feature_dim, 1),
            )

        self._tensors["hidden"] = _alloc("hidden", hidden_dim)
        self._tensors["full_k"] = _alloc("full_k", kv_num_heads * full_head_dim)
        self._tensors["full_v"] = _alloc("full_v", kv_num_heads * full_head_dim)
        self._tensors["slid_k"] = _alloc("slid_k", kv_num_heads * slid_head_dim)
        self._tensors["slid_v"] = _alloc("slid_v", kv_num_heads * slid_head_dim)

    def append(
        self,
        hidden: torch.Tensor,       # [L, hidden_dim]
        full_k: torch.Tensor,       # [kv_num_heads, L, full_head_dim]  OR [L, kv*full_head_dim]
        full_v: torch.Tensor,
        slid_k: torch.Tensor,
        slid_v: torch.Tensor,
    ) -> None:
        if self._n >= self.num_samples:
            raise RuntimeError("Writer received more samples than expected")
        L = int(hidden.shape[0])
        end = self._pos + L

        def _flat(x, per_head_dim):
            # Accept [kv, L, per_head_dim] or [L, kv*per_head_dim].
            if x.ndim == 3:
                # [kv, L, d] -> [L, kv*d]
                x = x.transpose(0, 1).contiguous().view(L, self.kv_num_heads * per_head_dim)
            elif x.ndim == 2:
                pass
            else:
                raise ValueError(f"Bad shape {tuple(x.shape)}")
            return x.to(self.dtype)

        self._tensors["hidden"][self._pos:end] = hidden.to(self.dtype)
        self._tensors["full_k"][self._pos:end] = _flat(full_k, self.full_head_dim)
        self._tensors["full_v"][self._pos:end] = _flat(full_v, self.full_head_dim)
        self._tensors["slid_k"][self._pos:end] = _flat(slid_k, self.slid_head_dim)
        self._tensors["slid_v"][self._pos:end] = _flat(slid_v, self.slid_head_dim)
        self._pos = end
        self._n += 1
        self.offsets[self._n] = end

    def set_meta(self, meta: Dict[str, Any]) -> None:
        self._meta = dict(meta)

    def close(self) -> None:
        if self._n != self.num_samples:
            raise RuntimeError(f"Expected {self.num_samples} samples, got {self._n}")
        if self._pos != self.total_tokens:
            raise RuntimeError(f"Expected {self.total_tokens} tokens, got {self._pos}")

        for field in self.FIELDS:
            torch.save({"flat": self._tensors[field]}, f"{self.output_path}.{field}")
            try:
                os.remove(self._raw_paths[field])
            except OSError:
                pass

        meta = dict(self._meta or {})
        meta["cache_format"] = "gemma_mtp_v1"
        meta["hidden_dim"] = self.hidden_dim
        meta["kv_num_heads"] = self.kv_num_heads
        meta["full_head_dim"] = self.full_head_dim
        meta["slid_head_dim"] = self.slid_head_dim
        meta["hidden_dtype"] = str(self.dtype).replace("torch.", "")
        meta["offsets"] = self.offsets
        torch.save(meta, self.output_path)


class LazyGemmaCache:
    """Gemma4-specific view with lazy per-sample access to hidden + KV states.

    __getitem__(i) returns a dict {'hidden': [L,d], 'full_k': [kv,L,dfull],
    'full_v': [kv,L,dfull], 'slid_k': [kv,L,dslid], 'slid_v': [kv,L,dslid]}.
    All returned tensors are zero-copy views into the mmap'd flat storages.
    """

    def __init__(self, flats: Dict[str, torch.Tensor], offsets: torch.Tensor,
                 kv_num_heads: int, full_head_dim: int, slid_head_dim: int):
        self._flats = flats
        self._offsets = offsets
        self._n = int(offsets.numel()) - 1
        self._kv = kv_num_heads
        self._df = full_head_dim
        self._ds = slid_head_dim

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._n)
            return [self[i] for i in range(start, stop, step)]
        if idx < 0:
            idx += self._n
        s = int(self._offsets[idx].item())
        e = int(self._offsets[idx + 1].item())
        L = e - s
        kv, df, ds = self._kv, self._df, self._ds
        return {
            "hidden": self._flats["hidden"][s:e],                    # [L, d]
            "full_k": self._flats["full_k"][s:e].view(L, kv, df).transpose(0, 1),  # [kv, L, df]
            "full_v": self._flats["full_v"][s:e].view(L, kv, df).transpose(0, 1),
            "slid_k": self._flats["slid_k"][s:e].view(L, kv, ds).transpose(0, 1),
            "slid_v": self._flats["slid_v"][s:e].view(L, kv, ds).transpose(0, 1),
        }


def load_cache(path: str, mmap: bool = False) -> Dict[str, Any]:
    """Load a cache in any supported format.

    Returns a dict where:
      - mmap_v1: cache["hidden_states"] is a LazyHiddenStates view.
      - gemma_mtp_v1: cache["samples"] is a LazyGemmaCache view.
      - legacy: cache["hidden_states"] is a python list.
    """
    meta = torch.load(path, map_location="cpu", weights_only=False)
    fmt = meta.get("cache_format")
    if fmt == "mmap_v1":
        hidden_path = path + ".hidden"
        if not os.path.exists(hidden_path):
            raise FileNotFoundError(f"Missing {hidden_path} for mmap_v1 cache")
        hidden_bundle = torch.load(hidden_path, map_location="cpu",
                                   weights_only=False, mmap=mmap)
        flat = hidden_bundle["hidden_flat"]
        offsets = meta["offsets"]
        meta["hidden_states"] = LazyHiddenStates(flat, offsets)
        return meta

    if fmt == "gemma_mtp_v1":
        flats: Dict[str, torch.Tensor] = {}
        for field in GemmaStreamingCacheWriter.FIELDS:
            fp = f"{path}.{field}"
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing {fp} for gemma_mtp_v1 cache")
            bundle = torch.load(fp, map_location="cpu", weights_only=False, mmap=mmap)
            flats[field] = bundle["flat"]
        meta["samples"] = LazyGemmaCache(
            flats, meta["offsets"],
            meta["kv_num_heads"], meta["full_head_dim"], meta["slid_head_dim"],
        )
        return meta

    # Legacy format: hidden_states is already a python list of tensors.
    return meta
