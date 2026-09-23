"""
Entry point: generate text from the fine-tuned GPT-2-medium checkpoint
produced by run_finetune_gpt2_medium.py.

Usage:
    python run_generate_gpt2_medium.py "What is DU error code 1234?"
"""

import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "gpt2-medium-finetuned"

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am"

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
