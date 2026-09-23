"""
The training loop: how the model actually learns from your documents.

At a high level, every training step does the same four things:
  1. Feed a batch of input token sequences into the model -> get logits.
  2. Compare the predicted next-token distribution (logits) against the
     actual next tokens (target) using cross-entropy loss.
  3. Backpropagate: compute how much each weight contributed to the loss.
  4. Nudge every weight slightly in the direction that reduces the loss
     (the optimizer step).

Repeated over millions of tokens, this is the entire mechanism by which the
model "learns" the statistical patterns of your training documents - nothing
more mysterious than gradient descent on next-token prediction.

Adapted from ch05/01_main-chapter-code/gpt_train.py.
"""

import tiktoken
import torch

from .config import GPT_CONFIG_SMALL
from .data_ingestion import load_training_corpus
from .dataset import create_dataloader
from .generate import generate_text_simple, text_to_token_ids, token_ids_to_text
from .model import GPTModel


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)  # (batch, T, vocab_size)
    # Cross-entropy compares the predicted distribution over vocab_size
    # tokens against the single correct next token, at every position of
    # every sequence in the batch (flattened into one big batch of
    # predictions for the loss function).
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Prints a short sample of what the model would generate right now -
    a quick, qualitative sanity check that runs after every epoch."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_size=context_size)
    print(" ", token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                        num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()              # clear gradients from the previous step
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                    # backpropagation: compute gradients
            optimizer.step()                   # apply the gradient update to every weight
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (step {global_step:06d}): "
                      f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        print(f"--- sample generation after epoch {epoch + 1} ---")
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def main(
    training_data_dir="training_data",
    cfg=None,
    learning_rate=5e-4,
    weight_decay=0.1,
    num_epochs=10,
    batch_size=2,
    train_ratio=0.9,
    start_context="",
    model_out_path="model.pth",
):
    cfg = cfg or GPT_CONFIG_SMALL
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Turn your Excel/PDF/Word/text files into one text corpus ---
    print(f"Loading training materials from '{training_data_dir}'...")
    text_data = load_training_corpus(training_data_dir)
    print(f"Loaded {len(text_data):,} characters of training text.")

    # --- 2. Build the model ---
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- 3. Split into train/validation and build dataloaders ---
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader(
        text_data[:split_idx], batch_size=batch_size,
        max_length=cfg["context_length"], stride=cfg["context_length"],
        drop_last=True, shuffle=True,
    )
    val_loader = create_dataloader(
        text_data[split_idx:], batch_size=batch_size,
        max_length=cfg["context_length"], stride=cfg["context_length"],
        drop_last=False, shuffle=False,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Training corpus is too short for the configured context_length "
            f"({cfg['context_length']} tokens). Add more training documents "
            "or lower context_length in config.py (e.g. GPT_CONFIG_SMALL)."
        )

    # --- 4. Train ---
    tokenizer = tiktoken.get_encoding("gpt2")
    if not start_context:
        # Seed the qualitative sample generations with the first few words
        # of your own corpus, so the sanity-check output is relevant to it.
        start_context = text_data[:50]

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context=start_context, tokenizer=tokenizer,
    )

    # --- 5. Save the trained weights ---
    torch.save(model.state_dict(), model_out_path)
    print(f"Saved trained model weights to '{model_out_path}'")

    return model, train_losses, val_losses, tokens_seen


if __name__ == "__main__":
    main()
