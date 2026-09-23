# gpt-2-model

A self-contained, heavily-annotated GPT-2 implementation, built by reusing
the tested code from `ch02`-`ch05` of this repository. The goal of this
folder is to let you read the code top-to-bottom and understand exactly how
a GPT model works end to end - and then train it on your own documents
(Excel, PDF, Word, or plain text).

## Folder layout

```
gpt-2-model/
├── gpt2/
│   ├── model.py           <- the architecture itself (start here)
│   ├── dataset.py         <- turns text into (input, target) training pairs
│   ├── data_ingestion.py  <- reads your .xlsx/.pdf/.docx files into text
│   ├── generate.py        <- turns model output back into text
│   ├── train.py           <- the training loop
│   └── config.py          <- model size presets, explained inline
├── training_data/         <- put your Excel/PDF/Word/text files here
├── run_training.py        <- entry point: trains a model on training_data/
├── run_generate.py        <- entry point: generate text from a trained model
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

1. Drop your training documents into `training_data/` - `.xlsx`/`.xls`,
   `.pdf`, `.docx`, `.txt`, `.md`, or `.csv` files are all picked up
   automatically and concatenated into one training corpus (see
   `gpt2/data_ingestion.py`). Only files directly inside the folder are
   read, not subfolders.
2. `python run_training.py` - trains a small model and saves `model.pth`.
3. `python run_generate.py "your prompt here"` - generates text from it.

`run_training.py` defaults to `GPT_CONFIG_SMALL` (a few million parameters)
so it trains reasonably on a laptop CPU. Switch to `GPT_CONFIG_124M` in
`gpt2/config.py` once you have more data and compute (e.g. a GPU) - it's the
real GPT-2 "small" size used by OpenAI's original checkpoint.

## How the whole thing fits together

```
Your documents (.xlsx/.pdf/.docx/.txt)
        │  data_ingestion.py: extract_text() per file type
        ▼
   one long text corpus
        │  dataset.py: tokenize (tiktoken "gpt2" BPE) + sliding window
        ▼
 (input_ids, target_ids) pairs   <- target is input shifted by one token
        │  train.py: DataLoader batches these
        ▼
  ┌───────────────────────────────────────────────┐
  │                  GPTModel                      │
  │                                                 │
  │  token_emb(input_ids) + pos_emb(positions)     │
  │             │                                  │
  │             ▼                                  │
  │  [ TransformerBlock ] × n_layers               │
  │      norm -> MultiHeadAttention -> +residual   │
  │      norm -> FeedForward        -> +residual   │
  │             │                                  │
  │             ▼                                  │
  │  final LayerNorm -> Linear(emb_dim, vocab_size)│
  └───────────────────────────────────────────────┘
        │
        ▼
   logits (batch, T, vocab_size)
        │  train.py: cross_entropy(logits, target_ids)  <- the training signal
        │  generate.py: argmax over the last position    <- at inference time
        ▼
  loss.backward() + optimizer.step()      or      next predicted token
```

## The core ideas, and where to find them in the code

**Every token is a vector of size `emb_dim` (a.k.a. `d_model`).** This size
is fixed once in `config.py` and never changes anywhere in the network -
`tok_emb`, `pos_emb`, every attention block, every feed-forward block, and
`out_head`'s input are all `emb_dim`-wide. This is required by the residual
connections in `TransformerBlock.forward` (`x = x + sublayer(x)`), which
only work if `sublayer(x)` has the same shape as `x`.

**`context_length` is unrelated to `emb_dim`.** It only controls how many
tokens the model can look back at (the size of the positional embedding
table and the causal mask), not the width of each token's vector. See the
comments at the top of `config.py`.

**Q, K, V play different roles** (see the `MultiHeadAttention` docstring in
`model.py`):
- Query & Key are dotted together (`Q @ K^T`) to produce a `(T, T)` matrix of
  "how much should token i attend to token j" scores.
- Value is never part of that scoring - it's the actual content that gets
  copied into the output, weighted by the scores above (`softmax(...) @ V`).

**Multi-head attention** splits the `emb_dim`-wide Q/K/V vectors into
`num_heads` chunks of `head_dim = emb_dim / num_heads` each, and runs
attention independently within each chunk. This lets different heads
specialize in different relationships (syntax, coreference, position, ...)
at no extra parameter cost - see the step-by-step comments in
`MultiHeadAttention.forward`.

**The causal mask** is what makes this GPT-style (autoregressive) rather
than BERT-style (bidirectional): it zeroes out (`-inf` before softmax) any
attention score pointing at a future token, so predictions only ever depend
on earlier tokens.

**Training = next-token prediction.** `dataset.py` builds `target` as
`input` shifted one token to the right. `train.py`'s `calc_loss_batch`
compares the model's predicted next-token distribution (`logits`) against
that target with cross-entropy loss, and standard backpropagation +
AdamW does the rest.

## Notes on your training materials

- `data_ingestion.py` only extracts *text*. Tables in Excel become
  space-joined rows; formatting, images, and charts are ignored - the model
  only ever learns from the words.
- Small text corpora (a few pages) will train fast but the model will
  mostly memorize/overfit rather than generalize - that's expected and fine
  for learning how the mechanics work. Real GPT-2 was trained on ~40GB of
  text.
- If you see `ValueError: Training corpus is too short...`, either add more
  documents or lower `context_length` in `GPT_CONFIG_SMALL`.
