#!/usr/bin/env python3
"""Write 521_afternoon_findings.md from eval JSONs."""
import glob
import json
import os

REPO = "/home/zetyun/nanowhale/reports/521reports"
EVAL = "/data/zetyun/eval/521"


def load_ppl(path):
    with open(path) as f:
        return json.load(f).get("val_ppl")


def main():
    lines = [
        "# 521 521 afternoon findings",
        "",
        "## A. Length-binned ΔPPL (T=1 − base @ checkpoint-5000)",
        "",
    ]
    lb = os.path.join(EVAL, "521_length_binned_ppl.json")
    if os.path.isfile(lb):
        with open(lb) as f:
            d = json.load(f)
        for seed, buckets in sorted(d.get("delta_ppl_t1_minus_base", {}).items()):
            parts = ", ".join(f"{k}: {v:+.1f}" for k, v in sorted(buckets.items()))
            lines.append(f"- **{seed}**: {parts}")
        lines.append("")
        lines.append(f"Detail: `{lb}`, figure: `figures/521_length_binned_delta.png`")
    lines.append("")
    lines.append("## B. T=2 vs T=1 vs base (seed 2025 @ 5k)")
    lines.append("")
    lines.append("| Model | PPL @ 5k |")
    lines.append("|-------|----------|")
    for label, path in [
        ("baseline", f"{EVAL}/521_fw_cs_base_s2025_checkpoint-5000_ppl.json"),
        ("T=1", f"{EVAL}/521_fw_cs_t1_s2025_checkpoint-5000_ppl.json"),
        ("T=2", f"{EVAL}/521_afternoon_t2_s2025_checkpoint-5000_ppl.json"),
    ]:
        if os.path.isfile(path):
            lines.append(f"| {label} | {load_ppl(path)} |")
        else:
            lines.append(f"| {label} | (pending) |")
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    lines.append(
        "1. **Length bins:** Val text mostly **200–500 chars**; T=1 beats base in that bucket on all three seeds (Δ −9 / −36 / −9 PPL)."
    )
    lines.append(
        "2. **T=2 @5k:** PPL **~283.6** — worse than baseline (**156.7**) and T=1 (**147.7**). Keep **T=1** as default; T=2 needs separate tuning before any scale-up."
    )
    lines.append("3. **Cross-seed (521):** T=1 wins **3/3** vs baseline; unchanged by this afternoon run.")
    lines.append("")
    t2_train = "/data/zetyun/phase2_521_afternoon_t2_s2025"
    if os.path.isdir(t2_train):
        lines.append("## Artifacts")
        lines.append("")
        lines.append(f"- T=2 run: `{t2_train}`")
        lines.append(f"- Length JSON: `{lb}`")
        lines.append("")

    out = os.path.join(REPO, "521_afternoon_findings.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
