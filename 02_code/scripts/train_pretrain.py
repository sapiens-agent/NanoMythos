"""Pretrain a DeepSeek-V4 model from scratch on FineWeb-Edu using TRL SFTTrainer.

Usage:
  python scripts/train_pretrain.py --config configs/main_100m.yaml [--debug]
  accelerate launch --num_processes 8 scripts/train_pretrain.py --config configs/main_100m.yaml
"""

import os
import sys
import argparse
import glob
import hashlib
import json
import math
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

# Register for Auto classes
AutoConfig.register("nanowhale_deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    """Build a DeepSeek-V4 model from scratch (random init)."""
    config = DeepseekV4Config(**model_cfg)
    model = DeepseekV4ForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    return model


def build_tokenizer(tokenizer_path="tokenizer"):
    """Load the DeepSeek-V4 tokenizer."""
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def resolve_parquet_files(train_parquet: str):
    """Resolve parquet inputs from file/dir/glob/comma-separated patterns."""
    parts = [p.strip() for p in train_parquet.split(",") if p.strip()]
    files = []
    for part in parts:
        if os.path.isdir(part):
            files.extend(sorted(glob.glob(os.path.join(part, "*.parquet"))))
        elif any(ch in part for ch in ["*", "?", "["]):
            files.extend(sorted(glob.glob(part)))
        else:
            files.append(part)

    files = [os.path.abspath(p) for p in files if p.endswith(".parquet")]
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No parquet files resolved from --train_parquet={train_parquet!r}")
    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        raise ValueError(f"Parquet files not found: {missing}")
    return files


def _audit_rows_sha256(dataset, text_field: str) -> str:
    """SHA256 over sorted ``row_id\\ttext`` lines for reproducibility checks."""
    lines = []
    for i in range(len(dataset)):
        row = dataset[i]
        tid = row.get("nanowhale_row_id", i)
        text = row.get(text_field, "") or ""
        lines.append(f"{tid}\t{text}")
    lines.sort()
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def apply_parquet_train_val_split(
    dataset,
    text_field: str,
    val_ratio: float,
    split_seed: int,
    split_method: str,
):
    """Return ``(train_dataset, val_dataset, mapped_all)`` for audit.

    ``mapped_all`` is the full table after ``nanowhale_row_id`` and ``nanowhale_is_val``
    assignment (used for checksum). Train/val drops ``nanowhale_is_val`` only.
    """
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if split_method not in ("hash", "row_order"):
        raise ValueError(f"split_method must be 'hash' or 'row_order', got {split_method!r}")

    n = len(dataset)
    row_ids = list(range(n))
    ds = dataset.add_column("nanowhale_row_id", row_ids)
    n_val = max(1, int(math.ceil(val_ratio * n)))
    threshold = int(round(val_ratio * 1_000_000))

    def mark_split(example, index):
        text = example.get(text_field, "") or ""
        text = text if isinstance(text, str) else str(text)
        if split_method == "row_order":
            is_val = index >= (n - n_val)
        else:
            payload = f"{split_seed}\t{index}\t{text}".encode("utf-8", errors="replace")
            h = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            is_val = (h % 1_000_000) < threshold
        return {"nanowhale_is_val": is_val}

    mapped_all = ds.map(mark_split, with_indices=True)
    val_ds = mapped_all.filter(lambda ex: ex["nanowhale_is_val"])
    train_ds = mapped_all.filter(lambda ex: not ex["nanowhale_is_val"])
    if len(train_ds) < 1:
        raise ValueError("Train split is empty; decrease val_ratio or check data.")
    drop_cols = [c for c in ("nanowhale_is_val",) if c in train_ds.column_names]
    train_ds = train_ds.remove_columns(drop_cols)
    val_ds = val_ds.remove_columns([c for c in ("nanowhale_is_val",) if c in val_ds.column_names])
    return train_ds, val_ds, mapped_all


def write_val_split_artifacts(
    val_dataset,
    train_dataset,
    full_dataset_for_audit,
    text_field: str,
    val_text_out: str,
    split_meta_out: str,
    parquet_files: list,
    val_ratio: float,
    split_seed: int,
    split_method: str,
    max_chars_per_line: int = 400,
):
    os.makedirs(os.path.dirname(val_text_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(split_meta_out) or ".", exist_ok=True)

    val_lines = []
    for i in range(len(val_dataset)):
        row = val_dataset[i]
        t = row.get(text_field, "") or ""
        t = t if isinstance(t, str) else str(t)
        t = t.strip().replace("\n", " ").replace("\r", " ")
        if len(t) > max_chars_per_line:
            t = t[:max_chars_per_line]
        val_lines.append(t)

    with open(val_text_out, "w", encoding="utf-8") as f:
        for ln in val_lines:
            f.write(ln + "\n")

    def char_stats(ds):
        lens = []
        for i in range(len(ds)):
            row = ds[i]
            t = row.get(text_field, "") or ""
            t = t if isinstance(t, str) else str(t)
            lens.append(len(t))
        if not lens:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0}
        sl = sorted(lens)
        mid = len(sl) // 2
        med = sl[mid] if len(sl) % 2 == 1 else 0.5 * (sl[mid - 1] + sl[mid])
        p95_idx = min(len(sl) - 1, int(math.ceil(0.95 * len(sl))) - 1)
        return {
            "mean": float(sum(sl) / len(sl)),
            "median": float(med),
            "p95": float(sl[p95_idx]),
        }

    meta = {
        "parquet_files": parquet_files,
        "parquet_text_field": text_field,
        "val_ratio": val_ratio,
        "split_seed": split_seed,
        "split_method": split_method,
        "n_total": len(full_dataset_for_audit),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "val_text_out": os.path.abspath(val_text_out),
        "split_meta_out": os.path.abspath(split_meta_out),
        "train_char_stats": char_stats(train_dataset),
        "val_char_stats": char_stats(val_dataset),
        "audit_sha256_sorted_rowid_text": _audit_rows_sha256(full_dataset_for_audit, text_field),
    }
    with open(split_meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


class NanowhaleDiagnosticsCallback(TrainerCallback):
    """Write ``model.model._diagnostics_last`` to ``diagnostics/steps.jsonl`` each step."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._fp = None
        self._path = None

    def on_train_begin(self, args, state, control, **kwargs):
        ddir = os.path.join(self.output_dir, "diagnostics")
        os.makedirs(ddir, exist_ok=True)
        self._path = os.path.join(ddir, "steps.jsonl")
        self._fp = open(self._path, "a", encoding="utf-8")

    def on_step_end(self, args, state, control, **kwargs):
        import json

        model = kwargs["model"]
        base = model.module if hasattr(model, "module") else model
        rec = getattr(base.model, "_diagnostics_last", None)
        if rec is None or self._fp is None:
            return
        row = {"step": int(state.global_step)}
        row.update(rec)
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main_100m.yaml")
    parser.add_argument("--debug", action="store_true", help="Use tiny subset for smoke testing")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--collect_diagnostics", action="store_true", help="Enable model diagnostic tensors (JSONL).")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training max_steps.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on CUDA.")
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Force fp32 training (override YAML training.bf16).",
    )
    parser.add_argument(
        "--synthetic_data",
        action="store_true",
        help="Use in-memory text only (no HuggingFace dataset download). For local smoke.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Trainer seed.")
    parser.add_argument(
        "--train_parquet",
        type=str,
        default=None,
        help="Local parquet source (file, directory, glob, or comma-separated list).",
    )
    parser.add_argument(
        "--parquet_text_field",
        type=str,
        default="text",
        help="Text column name in local parquet dataset.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="If >0 and --train_parquet is set, hold out this fraction for val (not used in training).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=2025,
        help="Seed for hash-based val assignment (independent of --seed).",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="hash",
        choices=("hash", "row_order"),
        help="hash: pseudo-random by row_id+text; row_order: last ceil(val_ratio*N) rows are val.",
    )
    parser.add_argument(
        "--val_text_out",
        type=str,
        default="/data/zetyun/eval/515_jd_val_split.txt",
        help="One text per line (val split only), written when --val_ratio > 0.",
    )
    parser.add_argument(
        "--split_meta_out",
        type=str,
        default="/data/zetyun/eval/516_split_meta.json",
        help="JSON metadata for the split, written when --val_ratio > 0.",
    )
    parser.add_argument(
        "--loop_delta_lambda",
        type=float,
        default=None,
        help="If set, overrides model.loop_delta_lambda in the YAML config before model init.",
    )
    parser.add_argument("--enable_coconut_light", action="store_true")
    parser.add_argument("--latent_steps", type=int, default=None)
    parser.add_argument("--latent_mode", type=str, default=None)
    parser.add_argument("--latent_loss_weight", type=float, default=None)
    parser.add_argument(
        "--use_recurrent_core_in_latent",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--detach_latent_state",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Tokenizer directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = dict(cfg["model"])
    train_cfg = cfg["training"]
    if args.loop_delta_lambda is not None:
        model_cfg["loop_delta_lambda"] = float(args.loop_delta_lambda)
    if args.enable_coconut_light:
        model_cfg["coconut_light_enabled"] = True
    if args.latent_steps is not None:
        model_cfg["coconut_latent_steps"] = int(args.latent_steps)
    if args.latent_mode is not None:
        model_cfg["coconut_latent_mode"] = args.latent_mode
    if args.latent_loss_weight is not None:
        model_cfg["coconut_latent_loss_weight"] = float(args.latent_loss_weight)
    if args.use_recurrent_core_in_latent is not None:
        model_cfg["coconut_use_recurrent_in_latent"] = bool(args.use_recurrent_core_in_latent)
    if args.detach_latent_state is not None:
        model_cfg["coconut_detach_latent"] = bool(args.detach_latent_state)
    if args.val_ratio and args.val_ratio > 0 and not args.train_parquet:
        raise ValueError("--val_ratio > 0 requires --train_parquet.")
    if args.collect_diagnostics:
        model_cfg["collect_hidden_states"] = True
        model_cfg["collect_moe_router_diagnostics"] = True
        model_cfg["collect_hyper_connection_diagnostics"] = True

    torch.set_float32_matmul_precision('high')
    tokenizer = build_tokenizer(args.tokenizer)
    for key, tok_key in (
        ("bot_token_id", "<|bot|>"),
        ("eot_token_id", "<|eot|>"),
        ("lat_token_id", "<|lat|>"),
    ):
        tid = tokenizer.convert_tokens_to_ids(tok_key)
        if tid is not None and tid != tokenizer.unk_token_id:
            model_cfg[key] = int(tid)
    if len(tokenizer) != model_cfg.get("vocab_size"):
        model_cfg["vocab_size"] = len(tokenizer)
    model = build_model(model_cfg)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available() and not args.no_compile:
        model = torch.compile(model)
        print("torch.compile enabled")
    elif args.no_compile:
        print("torch.compile disabled (--no_compile)")

    # Output directory
    output_dir = args.output_dir or f"checkpoints/pretrain_{os.path.basename(args.config).replace('.yaml', '')}"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    if args.train_parquet:
        parquet_files = resolve_parquet_files(args.train_parquet)
        dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
        )
        max_steps = train_cfg.get("max_steps", 20000)
        save_steps = train_cfg.get("save_steps", 2000)
        logging_steps = train_cfg.get("logging_steps", 10)
        if args.val_ratio and args.val_ratio > 0:
            train_ds, val_ds, mapped_all = apply_parquet_train_val_split(
                dataset,
                args.parquet_text_field,
                args.val_ratio,
                args.split_seed,
                args.split_method,
            )
            meta = write_val_split_artifacts(
                val_ds,
                train_ds,
                mapped_all,
                args.parquet_text_field,
                args.val_text_out,
                args.split_meta_out,
                parquet_files,
                args.val_ratio,
                args.split_seed,
                args.split_method,
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "split_meta.json"), "w", encoding="utf-8") as sf:
                json.dump(meta, sf, indent=2)
                sf.write("\n")
            dataset = train_ds
            if "nanowhale_row_id" in dataset.column_names:
                dataset = dataset.remove_columns(["nanowhale_row_id"])
            print(
                f"Train/val split: method={args.split_method} seed={args.split_seed} "
                f"ratio={args.val_ratio} -> n_train={meta['n_train']} n_val={meta['n_val']}"
            )
            print(f"Wrote val texts: {args.val_text_out}")
            print(f"Wrote split meta: {args.split_meta_out} (copy under {output_dir}/split_meta.json)")
    elif args.synthetic_data:
        from datasets import Dataset

        base = [
            "Nanowhale synthetic smoke sentence A. ",
            "Nanowhale synthetic smoke sentence B. ",
            "Nanowhale synthetic smoke sentence C. ",
        ]
        texts = [(b * 40) for b in base for _ in range(100)]
        dataset = Dataset.from_dict({"text": texts})
        max_steps = train_cfg.get("max_steps", 20)
        save_steps = max(1, max_steps)
        logging_steps = 1
    elif args.debug:
        # Tiny subset for smoke testing
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        dataset = dataset.take(200)
        # Convert to regular dataset
        from datasets import Dataset
        rows = list(dataset)
        dataset = Dataset.from_list(rows)
        max_steps = train_cfg.get("max_steps", 50)
        save_steps = 10
        logging_steps = 1
    else:
        # Stream the full dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        max_steps = train_cfg.get("max_steps", 20000)
        save_steps = train_cfg.get("save_steps", 2000)
        logging_steps = train_cfg.get("logging_steps", 10)

    if args.max_steps is not None:
        max_steps = args.max_steps

    if args.train_parquet and max_steps and save_steps > max_steps:
        save_steps = max_steps

    max_seq_length = train_cfg.get("max_seq_length", 2048)

    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    num_workers = 0 if args.debug or args.synthetic_data or not torch.cuda.is_available() else 4

    # SFT Config for pretraining (raw text, no chat template)
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        packing=True,
        dataset_text_field=(args.parquet_text_field if args.train_parquet else "text"),
        # Training
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 6e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        adam_beta1=train_cfg.get("adam_beta1", 0.9),
        adam_beta2=train_cfg.get("adam_beta2", 0.95),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        max_steps=max_steps,
        bf16=torch.cuda.is_available() and train_cfg.get("bf16", True) and not args.no_bf16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging
        logging_steps=logging_steps,
        logging_first_step=True,
        disable_tqdm=True,
        report_to=["none"],
        # Saving
        save_steps=save_steps,
        save_total_limit=3,
        # Hub
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,
        # Misc
        optim=optim_name,
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=(2 if num_workers > 0 else None),
        seed=args.seed,
        # DDP: MoE has unused params (inactive experts)
        ddp_find_unused_parameters=True,
    )

    # Initialize trackio
    try:
        import trackio
        trackio.init(
            project="smol-deepseek-v4",
            name=f"pretrain-dsv4-{os.path.basename(args.config).replace('.yaml', '')}",
        )
        sft_config.report_to = ["trackio"]
        print("Trackio logging enabled")
    except Exception as e:
        print(f"Trackio not available: {e}")

    callbacks = []
    if args.collect_diagnostics:
        callbacks.append(NanowhaleDiagnosticsCallback(output_dir))

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    # Print training info
    print(f"\n{'='*60}")
    print(f"Pretraining DeepSeek-V4")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} GA")
    print(f"Seq length: {sft_config.max_length}")
    print(f"LR: {sft_config.learning_rate}")
    print(f"Debug mode: {args.debug}")
    if args.train_parquet:
        print(f"Train parquet: {args.train_parquet}")
        print(f"Parquet text field: {args.parquet_text_field}")
    print(f"{'='*60}\n")

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final checkpoint
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nFinal model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
