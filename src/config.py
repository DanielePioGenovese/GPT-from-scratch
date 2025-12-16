import os

import tiktoken

# main.py
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# dataset
NUM_WORKERS = min(os.cpu_count(), 4)
DATA_DIR = "./data"
BATCH_SIZE = 64
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
OUTPUT_DIM = 6
