#!/usr/bin/env python3
"""Summarize nanowhale JSONL diagnostics and optionally emit plots."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def series_from_rows(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, list) and v:
            out.append(float(sum(v) / len(v)))
        elif isinstance(v, (int, float)):
            out.append(float(v))
    return out


def layerwise_mean_lists(rows: List[Dict[str, Any]], key: str) -> List[List[float]]:
    return [r[key] for r in rows if isinstance(r.get(key), list)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=str, help="Path to diagnostics/steps.jsonl")
    ap.add_argument("--out-md", type=str, required=True, help="Output markdown path")
    ap.add_argument("--out-png-hidden", type=str, default=None)
    ap.add_argument("--out-png-moe", type=str, default=None)
    ap.add_argument("--out-png-hc", type=str, default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)
    lines = []
    lines.append("# nanowhale diagnostics summary\n")
    lines.append(f"- source: `{args.jsonl}`\n")
    lines.append(f"- rows: {len(rows)}\n")

    if not rows:
        lines.append("\nNo rows; nothing to plot.\n")
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return

    sample = {k: v for k, v in rows[-1].items() if k != "step"}
    lines.append("\n## Last row keys (sample)\n\n```json\n")
    lines.append(json.dumps(sample, indent=2)[:4000])
    lines.append("\n```\n")

    # Plots (optional)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        lines.append("\n*matplotlib not installed; skipped PNG generation*\n")

    steps = [int(r["step"]) for r in rows if "step" in r]

    if plt is not None and args.out_png_hidden:
        hn = layerwise_mean_lists(rows, "hidden_norm_per_layer")
        if hn:
            plt.figure(figsize=(6, 4))
            for i, y in enumerate(hn):
                plt.plot(range(len(y)), y, marker="o", label=f"step {steps[i]}")
            plt.xlabel("layer")
            plt.ylabel("mean ||h|| (pooled)")
            plt.title("Hidden norm per layer")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(args.out_png_hidden)
            plt.close()
            lines.append(f"\n![hidden]({os.path.basename(args.out_png_hidden)})\n")

    if plt is not None and args.out_png_moe:
        ent = series_from_rows(rows, "moe_router_entropy")
        if ent:
            plt.figure(figsize=(6, 3))
            plt.plot(steps[: len(ent)], ent, marker="o")
            plt.xlabel("step")
            plt.ylabel("mean router entropy")
            plt.title("MoE router entropy (mean over layers)")
            plt.tight_layout()
            plt.savefig(args.out_png_moe)
            plt.close()
            lines.append(f"\n![moe]({os.path.basename(args.out_png_moe)})\n")

    if plt is not None and args.out_png_hc:
        ha = layerwise_mean_lists(rows, "hc_var_attn_in")
        if ha:
            plt.figure(figsize=(6, 4))
            for i, y in enumerate(ha):
                plt.plot(range(len(y)), y, marker="o", label=f"step {steps[i]}")
            plt.xlabel("layer")
            plt.ylabel("var across HC copies (attn in)")
            plt.title("Hyper-Connection copy variance")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(args.out_png_hc)
            plt.close()
            lines.append(f"\n![hc]({os.path.basename(args.out_png_hc)})\n")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
