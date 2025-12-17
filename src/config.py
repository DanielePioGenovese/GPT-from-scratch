import os

import tiktoken

# main.py
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# dataset
NUM_WORKERS = min(os.cpu_count(), 4)
DATA_DIR = "./data"
BATCH_SIZE = 4
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
OUTPUT_DIM = 6
MAX_LENGTH = 256
EMBED_DIM = 6

# model
NUM_HEADS = 2
NUM_LAYERS = 4
QKV_BIAS = False
DROPOUT_RATE = 0.1