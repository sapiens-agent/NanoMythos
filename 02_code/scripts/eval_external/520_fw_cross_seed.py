"""FineWeb-Edu cross-seed: baseline + T=1, seeds 2027, 2048."""
import sys, os, time, json
sys.path.insert(0, '/home/zetyun/nanowhale')
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from trl import SFTTrainer, SFTConfig
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

FW10K = '/data/zetyun/datasets/fineweb_edu_10k.parquet'
TOK_PATH = '/home/zetyun/nanowhale/tokenizer'
EVAL_DIR = '/data/zetyun/eval/520'

def build_model(rec_enabled, rec_steps):
    cfg = DeepseekV4Config(
        vocab_size=129280, hidden_size=320, num_hidden_layers=8,
        num_attention_heads=8, num_key_value_heads=1, moe_intermediate_size=640,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        q_lora_rank=160, head_dim=96, qk_rope_head_dim=32,
        o_groups=2, o_lora_rank=80, hc_mult=4, hc_sinkhorn_iters=2, hc_eps=1e-6,
        num_hash_layers=0, swiglu_limit=0.0, scoring_func='sqrtsoftplus',
        routed_scaling_factor=1.5, max_position_embeddings=2048, rms_norm_eps=1e-6,
        rope_theta=10000.0, initializer_range=0.02, tie_word_embeddings=False,
        attention_bias=False, attention_dropout=0.0, compress_ratios=[0]*9,
        hidden_smoothness_lambda=0.0, hidden_norm_lambda=0.0,
        recurrent_enabled=rec_enabled, recurrent_prelude_layers=2,
        recurrent_core_layers=4, recurrent_steps=rec_steps,
        recurrent_coda_layers=2, recurrent_use_loop_embedding=True,
        recurrent_max_steps=8,
    )
    return DeepseekV4ForCausalLM(cfg)

def run_heldout_ppl(ckpt_dir, val_text):
    tok = PreTrainedTokenizerFast.from_pretrained(ckpt_dir)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = DeepseekV4ForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float32)
    model = model.to('cuda'); model.eval()
    with open(val_text) as f: texts = [l.strip() for l in f if l.strip()]
    total_nll, total_toks = 0.0, 0
    with torch.no_grad():
        for text in texts:
            ids = tok.encode(text, return_tensors='pt').to('cuda')
            if ids.shape[1] < 2: continue
            out = model(ids, labels=ids)
            total_nll += out.loss.item() * (ids.shape[1] - 1)
            total_toks += ids.shape[1] - 1
    mean_nll = total_nll / max(total_toks, 1)
    return {'val_nll': round(mean_nll,4), 'val_ppl': round(float(torch.exp(torch.tensor(mean_nll)).item()), 1)}

def train_one(config_name, seed, output_dir, val_text_eval):
    rec_enabled = 't1' in config_name
    rec_steps = 1 if rec_enabled else 1
    label = 'T=1' if rec_enabled else 'Baseline'

    print("\n" + "="*60)
    print("Cross-seed: {} seed={}".format(label, seed))
    print("="*60)

    model = build_model(rec_enabled, rec_steps)
    n = sum(p.numel() for p in model.parameters())
    print("Params: {:,} ({:.1f}M)".format(n, n/1e6))

    tok = PreTrainedTokenizerFast.from_pretrained(TOK_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    torch.set_float32_matmul_precision('high')
    dataset = load_dataset('parquet', data_files={'train': FW10K}, split='train')
    os.makedirs(output_dir, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=output_dir, max_length=512, packing=True, dataset_text_field='text',
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        learning_rate=6e-4, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95,
        max_grad_norm=1.0, lr_scheduler_type='cosine', warmup_ratio=0.03,
        max_steps=2000, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=10, logging_first_step=True, disable_tqdm=True,
        report_to=['none'], save_steps=2000, save_total_limit=2,
        optim='adamw_torch_fused', dataloader_num_workers=4, seed=seed,
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset, processing_class=tok)
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'final'))
    tok.save_pretrained(os.path.join(output_dir, 'final'))

    # Held-out PPL
    for ckpt in ['checkpoint-2000', 'final']:
        ckpt_dir = os.path.join(output_dir, ckpt)
        if not os.path.isdir(ckpt_dir): continue
        r = run_heldout_ppl(ckpt_dir, val_text_eval)
        r['config'] = label; r['seed'] = seed; r['ckpt'] = ckpt
        json_out = os.path.join(EVAL_DIR, '520_fw_cs_{}_s{}_{}_ppl.json'.format(config_name, seed, ckpt))
        with open(json_out, 'w') as f: json.dump(r, f)
        print("  {} seed={} {} PPL={:.1f}".format(label, seed, ckpt, r['val_ppl']))

for seed in [2027, 2048]:
    train_one('fw_base', seed,
              '/data/zetyun/phase2_520_fw_cs_base_s{}'.format(seed),
              '/data/zetyun/eval/520/val_fw_cs_base_s{}.txt'.format(seed))
    train_one('fw_t1', seed,
              '/data/zetyun/phase2_520_fw_cs_t1_s{}'.format(seed),
              '/data/zetyun/eval/520/val_fw_cs_t1_s{}.txt'.format(seed))

print("\nFineWeb cross-seed complete!")
