"""521: FineWeb 10K cross-seed baseline + T=1, 5000 steps, staged held-out PPL."""
import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, "/home/zetyun/nanowhale")
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

FW10K = "/data/zetyun/datasets/fineweb_edu_10k.parquet"
VAL_TEXT = "/data/zetyun/eval/520/val_fw_base_s2025.txt"
EVAL_521 = "/data/zetyun/eval/521"
TOK_PATH = "/home/zetyun/nanowhale/tokenizer"
CKPT_NAMES = [
    "checkpoint-1000",
    "checkpoint-2000",
    "checkpoint-3000",
    "checkpoint-4000",
    "checkpoint-5000",
    "final",
]


def build_model(rec_enabled: bool):
    cfg = DeepseekV4Config(
        vocab_size=129280,
        hidden_size=320,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=1,
        moe_intermediate_size=640,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=160,
        head_dim=96,
        qk_rope_head_dim=32,
        o_groups=2,
        o_lora_rank=80,
        hc_mult=4,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        num_hash_layers=0,
        swiglu_limit=0.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        compress_ratios=[0] * 9,
        hidden_smoothness_lambda=0.0,
        hidden_norm_lambda=0.0,
        recurrent_enabled=rec_enabled,
        recurrent_prelude_layers=2,
        recurrent_core_layers=4,
        recurrent_steps=1,
        recurrent_coda_layers=2,
        recurrent_use_loop_embedding=True,
        recurrent_max_steps=8,
    )
    return DeepseekV4ForCausalLM(cfg)


def run_heldout_ppl(ckpt_dir: str, val_text: str, json_out: str | None = None):
    t0 = time.time()
    tok = PreTrainedTokenizerFast.from_pretrained(ckpt_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = DeepseekV4ForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float32)
    model = model.to("cuda")
    model.eval()
    with open(val_text) as f:
        texts = [line.strip() for line in f if line.strip()]
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for text in texts:
            ids = tok.encode(text, return_tensors="pt").to("cuda")
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_nll += out.loss.item() * n
            total_toks += n
    mean_nll = total_nll / max(total_toks, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    r = {
        "val_nll": round(mean_nll, 4),
        "val_ppl": round(ppl, 1),
        "eval_runtime_sec": round(time.time() - t0, 1),
    }
    if json_out:
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(r, f, indent=2)
    return r


def staged_eval(output_dir: str, label: str, seed: int):
    for ckpt_name in CKPT_NAMES:
        ckpt_dir = os.path.join(output_dir, ckpt_name)
        if not os.path.isdir(ckpt_dir):
            continue
        json_out = os.path.join(
            EVAL_521, f"521_fw_cs_{label}_s{seed}_{ckpt_name}_ppl.json"
        )
        r = run_heldout_ppl(ckpt_dir, VAL_TEXT, json_out)
        with open(json_out) as f:
            data = json.load(f)
        data.update(
            {
                "seed": seed,
                "model": label,
                "ckpt": ckpt_name,
                "data": "fineweb_edu_10k",
                "checkpoint_dir": ckpt_dir,
            }
        )
        with open(json_out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  eval {label} seed={seed} {ckpt_name} PPL={r['val_ppl']}")


def train_one(label: str, rec_enabled: bool, seed: int, resume_from: str | None = None):
    output_dir = f"/data/zetyun/phase2_521_fw_cs_{label}_s{seed}"
    print("\n" + "=" * 60)
    print(f"Train: {label} seed={seed} -> {output_dir}")
    print("=" * 60)

    if resume_from and os.path.isdir(resume_from):
        print(f"Resume from {resume_from}")
        model = DeepseekV4ForCausalLM.from_pretrained(
            resume_from, torch_dtype=torch.float32
        )
    else:
        model = build_model(rec_enabled)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,} ({n/1e6:.1f}M)")

    tok = PreTrainedTokenizerFast.from_pretrained(TOK_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.set_float32_matmul_precision("high")
    dataset = load_dataset("parquet", data_files={"train": FW10K}, split="train")
    os.makedirs(output_dir, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        max_length=512,
        packing=True,
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=6e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=5000,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=True,
        report_to=["none"],
        save_steps=1000,
        save_total_limit=6,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        seed=seed,
        ddp_find_unused_parameters=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(os.path.join(output_dir, "final"))
    tok.save_pretrained(os.path.join(output_dir, "final"))
    staged_eval(output_dir, label, seed)


def eval_only_legacy(output_dir: str, label: str, seed: int):
    print(f"Eval-only: {output_dir}")
    staged_eval(output_dir, label, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("all", "train", "eval"),
        default="all",
        help="all=train missing + eval; train=train only; eval=eval only",
    )
    p.add_argument("--seeds", type=str, default="2025,2027,2048")
    args = p.parse_args()
    os.makedirs(EVAL_521, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    jobs = []
    for seed in seeds:
        for label, rec in [("base", False), ("t1", True)]:
            out = f"/data/zetyun/phase2_521_fw_cs_{label}_s{seed}"
            jobs.append((label, rec, seed, out))

    for label, rec, seed, out in jobs:
        if seed == 2025:
            legacy = f"/data/zetyun/phase2_520_fineweb_fw_{label}_5k".replace(
                "fw_base_5k", "fw_base_5k"
            )
            if label == "base":
                legacy = "/data/zetyun/phase2_520_fineweb_fw_base_5k"
            else:
                legacy = "/data/zetyun/phase2_520_fineweb_fw_t1_5k"
            if args.mode in ("all", "eval"):
                eval_only_legacy(legacy, label, seed)
            continue

        final_ckpt = os.path.join(out, "checkpoint-5000")
        if os.path.isdir(final_ckpt) and args.mode == "eval":
            eval_only_legacy(out, label, seed)
            continue
        if os.path.isdir(os.path.join(out, "final")) and args.mode == "eval":
            eval_only_legacy(out, label, seed)
            continue

        if args.mode in ("all", "train"):
            resume = None
            if seed == 2027 and label == "base":
                c2k_520 = "/data/zetyun/phase2_520_fw_cs_base_s2027/checkpoint-2000"
                c2k_521 = os.path.join(out, "checkpoint-2000")
                if os.path.isdir(c2k_521):
                    resume = c2k_521
                elif os.path.isdir(c2k_520) and not os.path.isdir(
                    os.path.join(out, "checkpoint-5000")
                ):
                    resume = c2k_520
            train_one(label, rec, seed, resume_from=resume)


if __name__ == "__main__":
    main()
