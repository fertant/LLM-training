"""
GPT-2 configuration presets.

Every number here answers a question we walked through in conversation:

- vocab_size:      how many distinct tokens the tokenizer can produce.
                    Sets the size of the token-embedding table AND the size
                    of the final output layer (one logit per possible token).
- context_length:  the maximum number of tokens the model can look back at
                    in one forward pass. Fixes the size of the positional
                    embedding table and the (context_length x context_length)
                    causal mask used inside attention. Independent of emb_dim.
- emb_dim:         "d_model" - the width of the vector that represents ONE
                    token. This value is threaded through the entire network
                    (token embeddings, positional embeddings, every
                    attention block, every feed-forward block, the final
                    output projection) because residual connections
                    (x = x + sublayer(x)) require every sublayer's output to
                    have the exact same width as its input.
- n_heads:         how many parallel attention "views" each layer computes.
                    head_dim = emb_dim / n_heads (must divide evenly).
- n_layers:        how many TransformerBlocks are stacked. More layers =
                    more rounds of mixing/refining information already
                    inside the same context_length window (depth of
                    reasoning, not depth of "how far back" attention reaches).
- drop_rate:       dropout probability, used for regularization.
- qkv_bias:        whether the Q/K/V linear projections include a bias term
                    (the original GPT-2 checkpoint does use bias here).
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # GPT-2's byte-pair-encoding vocabulary (tiktoken "gpt2")
    "context_length": 1024,  # max sequence length the model was designed for
    "emb_dim": 768,          # d_model
    "n_heads": 12,           # head_dim = 768 / 12 = 64
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,        # matches the original OpenAI GPT-2 checkpoint
}

# A much smaller config, useful for quickly training/debugging on a laptop
# CPU with your own Excel/PDF/Word documents before scaling up.
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 192,          # d_model
    "n_heads": 6,            # head_dim = 192 / 6 = 32
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": True,
}
