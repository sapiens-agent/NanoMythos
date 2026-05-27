"""Copy/rename 520 FineWeb 5k staged PPL JSONs to 521 cross-seed naming (seed 2025)."""
import json
import os
import shutil

EVAL_520 = "/data/zetyun/eval/520"
EVAL_521 = "/data/zetyun/eval/521"
os.makedirs(EVAL_521, exist_ok=True)

MAPPING = [
    ("520_fw_base_5k", "base", "phase2_520_fineweb_fw_base_5k"),
    ("520_fw_t1_5k", "t1", "phase2_520_fineweb_fw_t1_5k"),
]

CKPTS = [
    "checkpoint-1000",
    "checkpoint-2000",
    "checkpoint-3000",
    "checkpoint-4000",
    "checkpoint-5000",
    "final",
]

for prefix_520, label, ckpt_root in MAPPING:
    for ckpt in CKPTS:
        src = os.path.join(EVAL_520, f"{prefix_520}_{ckpt}_ppl.json")
        dst = os.path.join(EVAL_521, f"521_fw_cs_{label}_s2025_{ckpt}_ppl.json")
        if not os.path.isfile(src):
            print(f"SKIP missing: {src}")
            continue
        with open(src) as f:
            r = json.load(f)
        r["seed"] = 2025
        r["model"] = label
        r["ckpt"] = ckpt
        r["data"] = "fineweb_edu_10k"
        r["checkpoint_dir"] = f"/data/zetyun/{ckpt_root}/{ckpt}"
        with open(dst, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Wrote {dst} PPL={r.get('val_ppl')}")

print("521 backfill s2025 done")
