"""
Turns raw text into (input, target) tensors GPTModel can train on.

Adapted from ch02/01_main-chapter-code (GPTDatasetV1 / create_dataloader_v1).

The task GPT is trained on is simple: given a chunk of tokens, predict the
NEXT token at every position. So for a tokenized text
    [t0, t1, t2, t3, t4, t5, ...]
one training example of length max_length=4 is:
    input  = [t0, t1, t2, t3]
    target = [t1, t2, t3, t4]      <- input shifted right by one

i.e. target[i] is always "the token that comes after input[i]". The model
never sees target directly; it only sees input, and the training loss
(see train.py) compares its predicted next-token distribution against
target at every position.

`stride` controls how much the sliding window moves between consecutive
examples. stride == max_length means no overlap (each token appears in
exactly one example); stride < max_length means examples overlap, which
gives you more training examples from the same text at the cost of some
redundancy.
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Slide a window of size max_length across the whole token stream.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128,
                       shuffle=True, drop_last=True, num_workers=0):
    """Tokenize `txt` with GPT-2's byte-pair encoding and wrap it in a
    PyTorch DataLoader that yields (input_batch, target_batch) pairs."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers,
    )
