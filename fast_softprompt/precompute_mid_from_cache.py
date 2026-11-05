"""Recompute one mid-layer hidden-state cache from an existing cache's token_ids.

Used for strict reproduction of the paper's MultiMax protocol: the paper reads
mid-layer hidden states at layers 8/16/20 for Qwen3.5-4B/9B/27B, while the
existing caches were built at layer 16/16/32.

Usage:
  python precompute_mid_from_cache.py \
    --cache cache_verih_4b.pt --model models/Qwen3.5-4B \
    --mid_layer 8 --output cache_verih_4b_mid8.pt
"""
import argparse
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/data1/suzeqin/SpeculativeProbing')
sys.path.insert(0, '/data1/suzeqin/SpeculativeProbing/fast_softprompt')
from cache_io import save_cache_mmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--mid_layer', type=int, required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='Debug: only first N samples')
    args = ap.parse_args()

    meta = torch.load(args.cache, map_location='cpu', weights_only=False)
    token_ids = meta['token_ids']
    labels = meta['labels']
    if args.limit:
        token_ids = token_ids[:args.limit]
        labels = labels[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    model.eval()
    device = str(model.lm_head.weight.device)
    pad_id = tokenizer.pad_token_id or 0

    hidden_states = []
    with torch.no_grad():
        for i in tqdm(range(0, len(token_ids), args.batch_size)):
            batch = token_ids[i:i + args.batch_size]
            max_len = max(len(x) for x in batch)
            padded = [x + [pad_id] * (max_len - len(x)) for x in batch]
            masks = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            out = model.model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
            h = out.hidden_states[args.mid_layer]
            for j, ids in enumerate(batch):
                hidden_states.append(h[j, :len(ids)].cpu().to(torch.bfloat16))

    cache = {k: v for k, v in meta.items()
             if k not in ('hidden_states', 'offsets', 'cache_format', 'hidden_dim', 'hidden_dtype')}
    cache['layer'] = args.mid_layer
    cache['num_train'] = meta['num_train'] if not args.limit else len(token_ids)
    cache['num_test'] = meta['num_test'] if not args.limit else 0
    cache['token_ids'] = token_ids
    cache['labels'] = labels
    cache['hidden_states'] = hidden_states
    save_cache_mmap(args.output, cache)
    print(f'Saved {args.output}: {len(hidden_states)} samples, layer={args.mid_layer}')


if __name__ == '__main__':
    main()
