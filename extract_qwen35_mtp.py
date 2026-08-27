#!/usr/bin/env python3
"""
Extract Qwen3.5 MTP tensors from a full sharded checkpoint into one safetensors file.

Usage:
  python extract_qwen35_mtp.py \
    --src "/share/shmatikov/collin/eagle_classifier/models/Qwen3.5-4B" \
    --dst "/share/shmatikov/collin/eagle_classifier/models/Qwen3.5-4B-mtp"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def extract_mtp(src_dir: Path, dst_dir: Path) -> None:
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    if not mtp_keys:
        raise RuntimeError("No `mtp.*` tensors found in source model index.")

    by_shard: dict[str, list[str]] = {}
    for key in mtp_keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    dst_dir.mkdir(parents=True, exist_ok=True)

    extracted = {}
    for shard_name, keys in sorted(by_shard.items()):
        shard_path = src_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard referenced by index: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                extracted[key] = f.get_tensor(key)

    out_weights = dst_dir / "mtp.safetensors"
    save_file(extracted, str(out_weights))

    manifest = {
        "source_model": str(src_dir),
        "source_index": str(index_path),
        "num_tensors": len(mtp_keys),
        "tensors": mtp_keys,
        "source_shards": sorted(by_shard.keys()),
        "output_file": str(out_weights),
    }
    with (dst_dir / "mtp_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Extracted {len(mtp_keys)} tensors")
    print(f"Wrote: {out_weights}")
    print(f"Wrote: {dst_dir / 'mtp_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen3.5 MTP tensors from full checkpoint.")
    parser.add_argument("--src", required=True, help="Source full model directory (contains model.safetensors.index.json)")
    parser.add_argument("--dst", required=True, help="Destination directory for MTP-only checkpoint")
    args = parser.parse_args()

    extract_mtp(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
