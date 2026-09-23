"""
Entry point: fine-tune the real, pretrained GPT-2-medium checkpoint (355M
params, from Hugging Face) on the files in ./training_data - pure PyTorch,
CPU-friendly, no quantization tooling (no llama.cpp / MLX / bitsandbytes).

Unlike run_training.py (which trains this repo's from-scratch GPTModel from
random weights), this script starts from OpenAI's actual GPT-2-medium
weights and continues training them on your documents, so it already knows
English and general language patterns before it ever sees your data.

Usage:
    python run_finetune_gpt2_medium.py

Requires (added to requirements.txt):
    pip install transformers

Notes for M1 CPU:
    - GPT-2-medium is ~1.4GB in fp32; AdamW keeps two extra states per
      parameter, so expect ~4-5GB of RAM used during training.
    - block_size=256 and batch_size=1 below are deliberately conservative
      for CPU. Raise them only if you have RAM/time to spare.
    - Full fine-tuning updates all 355M parameters. If this is too slow,
      the cheaper alternative is LoRA (freezes the base weights, trains
      small adapter matrices) - ask if you want that version instead.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from gpt2.data_ingestion import load_training_corpus

MODEL_NAME = "gpt2-medium"
OUT_DIR = "gpt2-medium-finetuned"


class BlockDataset(Dataset):
    """Chops one long token stream into fixed-size, non-overlapping blocks.

    GPT2LMHeadModel computes the next-token loss internally when given
    labels=input_ids (it shifts them by one position itself), so each
    training example here is just one block used as both input and label.
    """

    def __init__(self, token_ids, block_size):
        self.blocks = [
            torch.tensor(token_ids[i:i + block_size])
            for i in range(0, len(token_ids) - block_size, block_size)
        ]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def main(
    training_data_dir="training_data",
    model_name=MODEL_NAME,
    block_size=256,
    batch_size=1,
    num_epochs=3,
    learning_rate=5e-5,
    out_dir=OUT_DIR,
):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading pretrained '{model_name}' (this downloads ~1.4GB on first run)...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.train()

    print(f"Loading training materials from '{training_data_dir}'...")
    text_data = load_training_corpus(training_data_dir)
    print(f"Loaded {len(text_data):,} characters of training text.")

    token_ids = tokenizer.encode(text_data)
    dataset = BlockDataset(token_ids, block_size)
    if len(dataset) == 0:
        raise ValueError(
            f"Training corpus is too short for block_size={block_size}. "
            "Add more training documents or lower block_size."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}, step {step}: loss {loss.item():.3f}")

        print(f"--- Epoch {epoch + 1} done, avg loss {total_loss / len(loader):.3f} ---")

    print(f"Saving fine-tuned model to '{out_dir}'...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
