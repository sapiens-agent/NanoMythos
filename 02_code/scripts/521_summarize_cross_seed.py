#!/usr/bin/env python3
"""Print cross-seed summary table from eval/521 JSONs."""
import csv
import glob
import json
import os

EVAL = "/data/zetyun/eval/521"
OUT_CSV = "/home/zetyun/nanowhale/reports/521reports/tables/521_fineweb10k_cross_seed.csv"


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(EVAL, "521_fw_cs_*_ppl.json"))):
        with open(path) as f:
            r = json.load(f)
        base = os.path.basename(path)
        # 521_fw_cs_{model}_s{seed}_{ckpt}_ppl.json
        rest = base.replace("521_fw_cs_", "").replace("_ppl.json", "")
        parts = rest.rsplit("_", 1)
        ckpt = parts[-1] if parts[-1].startswith("checkpoint") or parts[-1] == "final" else "final"
        left = parts[0] if len(parts) > 1 else rest
        sp = left.rsplit("_s", 1)
        seed = int(sp[1]) if len(sp) == 2 else 0
        model = sp[0]
        rows.append(
            {
                "seed": seed,
                "model": model,
                "ckpt": ckpt,
                "val_nll": r.get("val_nll"),
                "val_ppl": r.get("val_ppl"),
                "eval_runtime_sec": r.get("eval_runtime_sec"),
            }
        )
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "model", "ckpt", "val_nll", "val_ppl", "eval_runtime_sec"],
        )
        w.writeheader()
        for row in sorted(rows, key=lambda x: (x["seed"], x["model"], x["ckpt"])):
            w.writerow(row)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
