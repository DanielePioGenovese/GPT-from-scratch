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
MAX_LENGTH = 1024
EMBED_DIM = 768

# model
NUM_HEADS = 12
NUM_LAYERS = 12
QKV_BIAS = False
MHA_DROPOUT_RATE = 0.1
FFN_DROPOUT_RATE = 0.1
EMB_DROPOUT_RATE = 0.1
LAYER_NORM_EPS = 1e-5
FF_HIDDEN_DIM = EMBED_DIM * 4