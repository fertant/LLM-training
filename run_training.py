"""
Entry point: train a GPT-2-style model on the files in ./training_data.

Usage:
    python run_training.py

Drop your .xlsx / .pdf / .docx / .txt files into the training_data/ folder
first (see training_data/README.md).
"""

from gpt2.config import GPT_CONFIG_SMALL
from gpt2.train import main

if __name__ == "__main__":
    main(
        training_data_dir="training_data",
        cfg=GPT_CONFIG_SMALL,   # swap for GPT_CONFIG_124M once you have more data/compute
        num_epochs=10,
        batch_size=2,
    )
