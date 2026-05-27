"""Build a reproducible document-level LM parquet for nanowhale Phase-2+ experiments.

Examples:
  python scripts/build_phase2_lm_parquet.py --source wikitext --max_rows 50000 --seed 42
  python scripts/build_phase2_lm_parquet.py --source tinystories --max_rows 20000 --seed 42
  python scripts/build_phase2_lm_parquet.py --source synthetic --max_rows 8000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=str,
        default="wikitext",
        choices=("wikitext", "tinystories", "fineweb", "synthetic"),
        help="HF dataset preset (`text` column), or `synthetic` for offline reproducible rows.",
    )
    p.add_argument("--max_rows", type=int, default=50_000, help="Max rows after filtering.")
    p.add_argument("--min_chars", type=int, default=50, help="Skip shorter texts after strip.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed before truncating.")
    p.add_argument(
        "--output",
        type=str,
        default="/data/zetyun/datasets/nanowhale_phase2_lm_train.parquet",
    )
    p.add_argument(
        "--manifest_out",
        type=str,
        default=None,
        help="Defaults to <output>.manifest.json",
    )
    p.add_argument(
        "--fineweb_take",
        type=int,
        default=100_000,
        help="When --source fineweb: stream then take this many before filter (repro cap).",
    )
    args = p.parse_args()
    manifest_out = args.manifest_out or (args.output + ".manifest.json")

    if args.source == "synthetic":
        name = "nanowhale/synthetic_lm_placeholder"
        rng = __import__("random").Random(args.seed)
        base = (
            "Nanowhale synthetic LM document for mechanism smoke. "
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models compress statistical regularities in text. "
        )
        rows = []
        for i in range(args.max_rows):
            noise = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(120))
            t = (base * 3 + noise).strip()
            assert len(t) >= args.min_chars
            rows.append({"text": t, "source": name, "source_row_id": i})
        ds = Dataset.from_list(rows)
    elif args.source == "wikitext":
        raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        name = "wikitext/wikitext-103-raw-v1"
        texts = []
        for i, row in enumerate(raw):
            t = (row.get("text") or "").strip()
            if len(t) < args.min_chars:
                continue
            texts.append({"text": t, "source": name, "source_row_id": i})
            if len(texts) >= args.max_rows * 2:
                break
        ds = Dataset.from_list(texts)
        ds = ds.shuffle(seed=args.seed).select(range(min(len(ds), args.max_rows)))
    elif args.source == "tinystories":
        raw = load_dataset("roneneldan/TinyStories", split="train")
        name = "roneneldan/TinyStories"
        texts = []
        for i, row in enumerate(raw):
            if i >= args.max_rows * 4:
                break
            t = (row.get("text") or "").strip()
            if len(t) < args.min_chars:
                continue
            texts.append({"text": t, "source": name, "source_row_id": i})
            if len(texts) >= args.max_rows:
                break
        ds = Dataset.from_list(texts)
        ds = ds.shuffle(seed=args.seed).select(range(min(len(ds), args.max_rows)))
    else:
        stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        name = "HuggingFaceFW/fineweb-edu"
        rows = []
        for i, row in enumerate(stream.take(args.fineweb_take)):
            t = (row.get("text") or "").strip()
            if len(t) < args.min_chars:
                continue
            rows.append({"text": t, "source": name, "source_row_id": i})
        ds = Dataset.from_list(rows)
        ds = ds.shuffle(seed=args.seed).select(range(min(len(ds), args.max_rows)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    ds.to_parquet(args.output)

    lens = [len(ds[i]["text"]) for i in range(len(ds))]
    lens.sort()
    mid = len(lens) // 2
    med = lens[mid] if len(lens) % 2 else 0.5 * (lens[mid - 1] + lens[mid])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "output_parquet": os.path.abspath(args.output),
        "n_rows": len(ds),
        "max_rows": args.max_rows,
        "min_chars": args.min_chars,
        "shuffle_seed": args.seed,
        "char_len_mean": sum(lens) / max(len(lens), 1),
        "char_len_median": float(med),
        "char_len_p95": float(lens[min(len(lens) - 1, int(0.95 * len(lens)) - 1)] if lens else 0),
        "license_note": (
            "synthetic: generated locally for reproducible offline smoke. "
            "Otherwise see Hugging Face dataset card (wikitext: CC BY-SA; TinyStories: Apache-2.0; FineWeb-Edu: ODC-BY)."
        ),
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))
    print("Wrote", args.output)
    print("Wrote", manifest_out)


if __name__ == "__main__":
    main()
