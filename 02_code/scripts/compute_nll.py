"""Mean NLL (nats/token) on fixed texts — small held-out sanity check."""

import argparse
import os
import sys

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


DEFAULT_TEXTS = [
    "The Earth orbits the Sun at a distance of approximately 93 million miles.",
    "Water is composed of two hydrogen atoms and one oxygen atom.",
    "Machine learning models generalize from data to unseen examples.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, default=None, help="Default: <model_path> or repo tokenizer/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--text_file", type=str, default=None, help="One sentence per line; overrides defaults")
    args = p.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = DeepseekV4ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    model = model.to(args.device)
    model.eval()

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = DEFAULT_TEXTS

    total_nll = 0.0
    total_toks = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt").to(args.device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_nll += out.loss.item() * n
            total_toks += n

    mean_nll = total_nll / max(total_toks, 1)
    print(f"mean_nll_nats_per_token: {mean_nll:.6f}")
    print(f"num_texts: {len(texts)}  total_label_tokens: {total_toks}")


if __name__ == "__main__":
    main()
