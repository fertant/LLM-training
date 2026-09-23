"""
The GPT-2 architecture, heavily annotated.

This file is adapted from ch04/01_main-chapter-code/gpt.py in this repo
(Sebastian Raschka, "Build a Large Language Model From Scratch"). The logic
is unchanged; the comments are expanded to explain WHY each piece exists and
how it connects to the others, so the whole forward pass can be read top to
bottom as one story.

Data flow through the whole model (see GPTModel.forward at the bottom):

    token ids (batch, T)
        -> token embedding + positional embedding      -> (batch, T, emb_dim)
        -> [ TransformerBlock ] x n_layers              -> (batch, T, emb_dim)
        -> final LayerNorm
        -> output linear layer (emb_dim -> vocab_size)  -> (batch, T, vocab_size)

Every tensor that flows between these stages has the SAME last dimension,
emb_dim (a.k.a. d_model). That is not a coincidence: TransformerBlock uses
residual ("skip") connections, x = x + sublayer(x), which only work if
sublayer(x) has the same shape as x.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention.

    Roles of Q, K, V (the most common point of confusion):
      - Query (Q) and Key (K) are used ONLY to decide "how much should token i
        pay attention to token j?". Their dot product Q @ K^T produces a
        (T, T) matrix of relevance scores, one row per token, one column per
        token it could attend to.
      - Value (V) is never involved in that scoring. It carries the actual
        content that gets copied/mixed into the output. The attention
        weights computed from Q/K are just the mixing coefficients applied
        to V: context = softmax(Q @ K^T) @ V.

    Multi-head split:
      Instead of running num_heads separate small attention modules, this
      class projects the input straight to the full emb_dim with ONE big
      W_query / W_key / W_value matrix each, then reshapes
      (batch, T, emb_dim) -> (batch, num_heads, T, head_dim) where
      head_dim = emb_dim // num_heads. Each head then computes its own
      (T, T) attention-score matrix independently, using only its own
      head_dim-wide slice of Q/K/V. This lets every head specialize in a
      different kind of relationship (syntax, coreference, position, ...)
      at no extra parameter cost, and lets PyTorch run all heads as one
      batched matmul instead of a Python loop.

    Causal mask:
      GPT is autoregressive - when predicting the next token, it must not
      "see" future tokens. `self.mask` is an upper-triangular matrix of 1s;
      wherever it's 1, we set the attention score to -inf BEFORE softmax, so
      after softmax those positions get exactly 0 weight. This is what
      makes the dependency structure specifically "each token can only
      depend on itself and earlier tokens".
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # width of Q/K/V per head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # learned mix of the concatenated heads
        self.dropout = nn.Dropout(dropout)

        # A fixed (not learned) buffer: upper-triangular mask of future positions.
        # Sliced down to (num_tokens, num_tokens) at forward time.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Step 1: project the input into query/key/value space.
        # Shape: (b, num_tokens, d_out) - still the FULL embedding width here.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Step 2: split the last dimension into (num_heads, head_dim). This is
        # a pure reshape - no computation - it just reinterprets which slice
        # of the d_out-wide vector belongs to which head.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 3: move the heads dimension next to batch so that every head
        # is treated as an independent "batch" for the matmuls that follow.
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Step 4: Q @ K^T for every head at once.
        # Shape: (b, num_heads, num_tokens, num_tokens) - one relevance matrix per head.
        attn_scores = queries @ keys.transpose(2, 3)

        # Step 5: apply the causal mask so no token can attend to future tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Step 6: turn scores into a probability distribution per row
        # (scaled by sqrt(head_dim) to keep gradients well-behaved - this is
        # the "scaled" in "scaled dot-product attention").
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 7: use the attention weights to take a weighted average of the
        # VALUE vectors - this is where information actually moves between
        # tokens. Q/K never appear again after this point.
        context_vec = (attn_weights @ values).transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)

        # Step 8: concatenate all heads back into one d_out-wide vector per
        # token (the inverse reshape of step 2), then let a learned linear
        # layer mix information across heads.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """
    Normalizes each token's vector to zero mean / unit variance (across the
    emb_dim axis, independently per token), then applies a learned
    scale/shift. This keeps activations in a well-behaved range as they flow
    through many stacked layers, which is essential for training deep
    transformers stably.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # avoids division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Smooth, non-linear activation function used inside the feed-forward
    block (GPT-2 uses this tanh-approximation instead of ReLU). Without a
    non-linearity here, stacking linear layers would collapse into a single
    linear transform no matter how many layers you stack.
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    A small per-token MLP: emb_dim -> 4*emb_dim -> emb_dim.

    Attention is the only place where tokens exchange information with each
    other. This block runs independently on each token's vector and gives
    the model extra capacity to transform/process what attention just
    gathered, before it's passed to the next layer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    One transformer layer = attention sub-block + feed-forward sub-block,
    each wrapped in "pre-norm + residual":

        x = x + Dropout(Attention(LayerNorm(x)))
        x = x + Dropout(FeedForward(LayerNorm(x)))

    The residual ("+ x") connections are why emb_dim must stay constant
    throughout the whole network - Attention(...) and FeedForward(...) must
    output something the same shape as their input so the addition works.
    Residuals also give gradients a direct path back to earlier layers
    during backpropagation, which is what makes stacking many layers
    (n_layers) trainable at all instead of vanishing/exploding.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sub-block, with its own residual connection.
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward sub-block, with its own residual connection.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    The full model: embeddings -> stacked TransformerBlocks -> output head.

    - tok_emb: one learned emb_dim-wide vector per vocabulary entry. This is
      the "meaning" of a token before any context is applied.
    - pos_emb: one learned emb_dim-wide vector per position (0 .. context_length-1).
      Attention itself has no notion of order (it just compares vectors), so
      position must be injected explicitly by adding this to the token
      embedding. This table's row count is exactly context_length, which is
      why the model cannot process a sequence longer than that.
    - trf_blocks: n_layers stacked TransformerBlocks, each refining the
      per-token representation using information gathered from other
      (earlier) tokens.
    - final_norm + out_head: project the final emb_dim-wide vector for each
      position into a vocab_size-wide vector of logits - the model's
      predicted "next token" distribution for that position.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (batch, seq_len, vocab_size)
        return logits

