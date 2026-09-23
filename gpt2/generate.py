"""
Turning the model's logits into actual generated text.

GPTModel.forward() returns logits of shape (batch, seq_len, vocab_size) - one
score per possible next token, at every position. `generate_text_simple`
repeatedly: runs the model, looks only at the LAST position's logits (that's
the prediction for "what comes after everything so far"), picks the
highest-scoring token (greedy decoding), appends it to the sequence, and
repeats. This is intentionally the simplest possible decoding strategy (no
temperature/top-k sampling) so the generation loop itself stays easy to
follow.
"""

import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension -> (1, T)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """idx: (batch, T) tensor of token ids to continue from."""
    for _ in range(max_new_tokens):
        # The model can only see up to context_size tokens at once (see
        # config.py's context_length) - crop older tokens if we exceed it.
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Only the last position's prediction matters for "the next token".
        logits = logits[:, -1, :]  # (batch, vocab_size)

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # greedy pick
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
