"""
Entry point: load trained weights from model.pth and generate text from a
prompt.

Usage:
    python run_generate.py "Once upon a time"
"""

import sys

import tiktoken
import torch

from gpt2.config import GPT_CONFIG_SMALL
from gpt2.generate import generate_text_simple, text_to_token_ids, token_ids_to_text
from gpt2.model import GPTModel

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am"
    cfg = GPT_CONFIG_SMALL

    model = GPTModel(cfg)
    model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = text_to_token_ids(prompt, tokenizer)

    out = generate_text_simple(model, encoded, max_new_tokens=50, context_size=cfg["context_length"])
    print(token_ids_to_text(out, tokenizer))
