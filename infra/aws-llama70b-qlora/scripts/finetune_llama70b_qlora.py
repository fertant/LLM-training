"""
QLoRA fine-tuning of a 70B Llama checkpoint (e.g. meta-llama/Llama-3.1-70B-*
or meta-llama/Llama-3.3-70B-Instruct) on the same training_data/ documents
used elsewhere in this repo.

Runs unattended on the EC2 instance this Terraform stack provisions (see
../ec2.tf / ../scripts/user_data.sh.tpl) - not meant to be run on a laptop.

QLoRA, in three parts:
  1. Load the 70B base model quantized to 4-bit (NF4) - shrinks it from
     ~140GB (fp16) down to ~35-40GB, small enough to shard across the
     4x24GB A10G GPUs on a g5.12xlarge.
  2. Freeze every one of those quantized weights and inject small trainable
     LoRA adapter matrices into the attention/MLP projections instead -
     only the adapters (tens of millions of params, not 70B) get gradients
     and optimizer states.
  3. Train normally; save just the adapter (a few hundred MB), not the
     70B base model, to output_dir.

This file is self-contained (duplicates the extension->reader mapping from
gpt2/data_ingestion.py) because it runs on a bare EC2 instance without the
rest of this repo installed.
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf", ".docx"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_xlsx(path: Path) -> str:
    import openpyxl

    workbook = openpyxl.load_workbook(path, data_only=True)
    lines = []
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            cells = [str(cell) for cell in row if cell is not None]
            if cells:
                lines.append(" ".join(cells))
    return "\n".join(lines)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


_READERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".csv": _read_txt,
    ".xlsx": _read_xlsx,
    ".xls": _read_xlsx,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_training_corpus(folder) -> str:
    folder = Path(folder)
    texts = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            texts.append(_READERS[path.suffix.lower()](path))

    if not texts:
        raise FileNotFoundError(
            f"No supported training files found in {folder}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return "<|endoftext|>".join(texts)


def build_block_dataset(tokenizer, text, block_size):
    token_ids = tokenizer.encode(text)
    blocks = [
        token_ids[i:i + block_size]
        for i in range(0, len(token_ids) - block_size, block_size)
    ]
    if not blocks:
        raise ValueError(
            f"Training corpus is too short for block_size={block_size}. "
            "Add more training documents or lower --block_size."
        )
    return Dataset.from_dict({"input_ids": blocks})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_id", required=True)
    parser.add_argument("--training_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer + 4-bit-quantized '{args.base_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",  # shards the quantized model across all visible GPUs
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Standard Llama attention + MLP projection names.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training materials from '{args.training_data_dir}'...")
    text_data = load_training_corpus(args.training_data_dir)
    print(f"Loaded {len(text_data):,} characters of training text.")

    dataset = build_block_dataset(tokenizer, text_data, args.block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()

    print(f"Saving LoRA adapter to '{args.output_dir}'...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
