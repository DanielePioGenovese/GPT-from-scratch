import tiktoken
from datasets import load_dataset
import numpy as np


def prepare_data():
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    print("Creating dataset_train.bin...")
    with open("dataset_train.bin", "wb") as f:
        for i, example in enumerate(ds):
            tokens = tokenizer.encode(
                example["text"], allowed_special={"<|endoftext|>"}
            )
            tokens.append(50256)  # <|endoftext|>
            f.write(np.array(tokens, dtype=np.uint16).tobytes())
            if i % 10000 == 0:
                print(f"Procecessed {i} stories...")
