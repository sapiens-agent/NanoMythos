#!/usr/bin/env python3
"""Per-document NLL by text-length bucket; compare base vs T=1 checkpoints."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_deepseek_v4 import DeepseekV4ForCausalLM
from transformers import PreTrainedTokenizerFast

VAL_TEXT = "/data/zetyun/eval/520/val_fw_base_s2025.txt"
EVAL_DIR = "/data/zetyun/eval/521"
REPO_OUT = "/home/zetyun/nanowhale/reports/521reports"

# (label, checkpoint dir)
DEFAULT_PAIRS = [
    ("base_s2025", "/data/zetyun/phase2_520_fineweb_fw_base_5k/checkpoint-5000"),
    ("t1_s2025", "/data/zetyun/phase2_520_fineweb_fw_t1_5k/checkpoint-5000"),
    ("base_s2027", "/data/zetyun/phase2_521_fw_cs_base_s2027/checkpoint-5000"),
    ("t1_s2027", "/data/zetyun/phase2_521_fw_cs_t1_s2027/checkpoint-5000"),
    ("base_s2048", "/data/zetyun/phase2_521_fw_cs_base_s2048/checkpoint-5000"),
    ("t1_s2048", "/data/zetyun/phase2_521_fw_cs_t1_s2048/checkpoint-5000"),
]

BUCKETS = [
    (0, 200, "0-200c"),
    (200, 500, "200-500c"),
    (500, 1000, "500-1kc"),
    (1000, 2000, "1k-2kc"),
    (2000, 10_000_000, "2kc+"),
]


def bucket_for(n_chars: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= n_chars < hi:
            return name
    return "other"


@torch.no_grad()
def per_text_nll(model, tok, texts, device):
    out = []
    for text in texts:
        ids = tok.encode(text, return_tensors="pt").to(device)
        if ids.shape[1] < 2:
            continue
        loss = model(ids, labels=ids).loss.item()
        n_tok = ids.shape[1] - 1
        out.append((len(text), n_tok, loss * n_tok, n_tok))
    return out


def aggregate(rows):
    """rows: list of (n_chars, n_tok, total_nll, n_label_toks)"""
    buckets = {}
    for n_chars, _n_tok, total_nll, n_label in rows:
        b = bucket_for(n_chars)
        if b not in buckets:
            buckets[b] = [0.0, 0]
        buckets[b][0] += total_nll
        buckets[b][1] += n_label
    result = {}
    for b, (tnll, nt) in buckets.items():
        result[b] = {
            "mean_nll": round(tnll / max(nt, 1), 4),
            "ppl": round(float(torch.exp(torch.tensor(tnll / max(nt, 1))).item()), 1),
            "n_tokens": nt,
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    with open(VAL_TEXT) as f:
        texts = [line.strip() for line in f if line.strip()]

    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPO_OUT, "tables"), exist_ok=True)

    all_models = {}
    for label, ckpt in DEFAULT_PAIRS:
        if not os.path.isdir(ckpt):
            print(f"SKIP {label}: missing {ckpt}")
            continue
        print(f"Eval length bins: {label}")
        tok = PreTrainedTokenizerFast.from_pretrained(ckpt)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = DeepseekV4ForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
        model = model.to(args.device)
        model.eval()
        rows = per_text_nll(model, tok, texts, args.device)
        all_models[label] = aggregate(rows)
        del model
        torch.cuda.empty_cache()

    # Pairwise delta t1 - base per seed
    deltas = {}
    for seed in ("2025", "2027", "2048"):
        b, t = f"base_s{seed}", f"t1_s{seed}"
        if b not in all_models or t not in all_models:
            continue
        deltas[seed] = {}
        for bk in all_models[b]:
            if bk in all_models[t]:
                deltas[seed][bk] = round(
                    all_models[t][bk]["ppl"] - all_models[b][bk]["ppl"], 1
                )

    out = {"per_model": all_models, "delta_ppl_t1_minus_base": deltas}
    json_path = os.path.join(EVAL_DIR, "521_length_binned_ppl.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")

    csv_path = os.path.join(REPO_OUT, "tables", "521_length_binned_delta.csv")
    bucket_names = [x[2] for x in BUCKETS]
    with open(csv_path, "w") as f:
        f.write("seed,bucket,delta_ppl_t1_minus_base\n")
        for seed, d in deltas.items():
            for bk in bucket_names:
                if bk in d:
                    f.write(f"{seed},{bk},{d[bk]}\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
