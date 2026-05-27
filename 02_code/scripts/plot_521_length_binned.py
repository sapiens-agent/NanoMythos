#!/usr/bin/env python3
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "/data/zetyun/eval/521/521_length_binned_ppl.json"
OUT = "/home/zetyun/nanowhale/reports/521reports/figures/521_length_binned_delta.png"
BUCKET_ORDER = ["0-200c", "200-500c", "500-1kc", "1k-2kc", "2kc+"]


def main():
    with open(JSON) as f:
        data = json.load(f)
    deltas = data.get("delta_ppl_t1_minus_base", {})
    seeds = sorted(deltas.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(BUCKET_ORDER)))
    w = 0.25
    for i, seed in enumerate(seeds):
        ys = [deltas[seed].get(b, 0) for b in BUCKET_ORDER]
        ax.bar([xi + (i - 1) * w for xi in x], ys, width=w, label=f"seed {seed}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER, rotation=15)
    ax.set_ylabel("Δ PPL (T=1 − base)")
    ax.set_title("FineWeb val: T=1 vs baseline by text length @ 5k")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
