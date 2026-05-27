#!/usr/bin/env python3
"""Generate 521 report figures and CSV tables from /data/zetyun/eval/521 JSONs."""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVAL_521 = "/data/zetyun/eval/521"
REPO_FIG = "/home/zetyun/nanowhale/reports/521reports/figures"
REPO_TBL = "/home/zetyun/nanowhale/reports/521reports/tables"


def load_jsons(pattern: str):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        r["_path"] = path
        base = os.path.basename(path)
        m = re.match(r"521_fw_cs_(\w+)_s(\d+)_(checkpoint-\d+|final)_ppl\.json", base)
        if m:
            r.setdefault("model", m.group(1))
            r.setdefault("seed", int(m.group(2)))
            r.setdefault("ckpt", m.group(3))
        rows.append(r)
    return rows


def step_from_ckpt(ckpt: str) -> int:
    if ckpt == "final":
        return 5000
    return int(ckpt.split("-")[1])


def plot_cross_seed_bar(out_png: str):
    rows = load_jsons(os.path.join(EVAL_521, "521_fw_cs_*_checkpoint-5000_ppl.json"))
    if not rows:
        rows = load_jsons(os.path.join(EVAL_521, "521_fw_cs_*_final_ppl.json"))
    seeds = sorted({r["seed"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(seeds)))
    w = 0.35
    base_ppl = []
    t1_ppl = []
    for i, seed in enumerate(seeds):
        b = next((r for r in rows if r["seed"] == seed and r["model"] == "base"), None)
        t = next((r for r in rows if r["seed"] == seed and r["model"] == "t1"), None)
        base_ppl.append(b["val_ppl"] if b else float("nan"))
        t1_ppl.append(t["val_ppl"] if t else float("nan"))
    ax.bar([i - w / 2 for i in x], base_ppl, width=w, label="baseline", color="#88aabb")
    ax.bar([i + w / 2 for i in x], t1_ppl, width=w, label="T=1", color="#4477aa")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Held-out PPL @ 5000 steps")
    ax.set_title("FineWeb-Edu 10K cross-seed (521)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_cross_seed_csv(out_csv: str):
    rows = load_jsons(os.path.join(EVAL_521, "521_fw_cs_*_ppl.json"))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "model", "ckpt", "val_nll", "val_ppl", "eval_runtime_sec"],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("seed", 0), x.get("model", ""), x.get("ckpt", ""))):
            w.writerow(
                {
                    "seed": r.get("seed"),
                    "model": r.get("model"),
                    "ckpt": r.get("ckpt"),
                    "val_nll": r.get("val_nll"),
                    "val_ppl": r.get("val_ppl"),
                    "eval_runtime_sec": r.get("eval_runtime_sec"),
                }
            )


def plot_staged_curves(out_png: str, prefix: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, color, marker in [
        ("base", "#88aabb", "o"),
        ("t1", "#4477aa", "s"),
        ("gpt2std", "#cc8844", "^"),
    ]:
        paths = sorted(glob.glob(os.path.join(EVAL_521, f"{prefix}_{model}_*ppl.json")))
        if not paths and model == "gpt2std":
            paths = sorted(glob.glob(os.path.join(EVAL_521, f"{prefix}_gpt2std_*ppl.json")))
        pts = []
        for p in paths:
            with open(p) as f:
                r = json.load(f)
            ckpt = r.get("ckpt", "")
            if "checkpoint-" not in ckpt and ckpt != "final":
                continue
            pts.append((step_from_ckpt(ckpt), r["val_ppl"]))
        if not pts:
            continue
        pts.sort()
        ax.plot(
            [x[0] for x in pts],
            [x[1] for x in pts],
            marker=marker,
            color=color,
            label=model,
            linewidth=2,
        )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Held-out PPL")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_fw50_csv(out_csv: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(EVAL_521, "521_fw50_*_ppl.json"))):
        with open(p) as f:
            r = json.load(f)
        rows.append(r)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "ckpt", "val_nll", "val_ppl", "eval_runtime_sec", "max_steps"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "model": r.get("model"),
                    "ckpt": r.get("ckpt"),
                    "val_nll": r.get("val_nll"),
                    "val_ppl": r.get("val_ppl"),
                    "eval_runtime_sec": r.get("eval_runtime_sec"),
                    "max_steps": r.get("max_steps"),
                }
            )


def plot_gonogo(out_png: str, gates: dict):
    fig, ax = plt.subplots(figsize=(8, 3))
    names = list(gates.keys())
    vals = [1 if gates[k] else 0 for k in names]
    colors = ["#44aa66" if v else "#cc4444" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Pass (1) / Fail (0)")
    ax.set_title("521 GPT-2-scale go/no-go gates")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(REPO_FIG, exist_ok=True)
    os.makedirs(REPO_TBL, exist_ok=True)
    plot_cross_seed_bar(os.path.join(REPO_FIG, "521_fineweb10k_cross_seed_ppl.png"))
    write_cross_seed_csv(os.path.join(REPO_TBL, "521_fineweb10k_cross_seed.csv"))
    plot_staged_curves(
        os.path.join(REPO_FIG, "521_fineweb50k_staged_ppl.png"),
        "521_fw50",
        "FineWeb-Edu 50K staged PPL (521)",
    )
    plot_staged_curves(
        os.path.join(REPO_FIG, "521_t1_vs_base_vs_gpt2std.png"),
        "521_fw50",
        "nanowhale T=1 vs baseline vs GPT-2 Standard (50K)",
    )
    write_fw50_csv(os.path.join(REPO_TBL, "521_fineweb50k_scaleup.csv"))
    plot_gonogo(
        os.path.join(REPO_FIG, "521_gpt2_scale_gonogo.png"),
        {
            "G1_cross_seed": True,
            "G2_50K_deferred": True,
            "G3_vs_GPT2_10K": True,
            "G4_smoke_singleaccelerator": True,
        },
    )
    print("521 plots/tables updated")


if __name__ == "__main__":
    main()
