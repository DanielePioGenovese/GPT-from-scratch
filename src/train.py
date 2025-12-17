import tiktoken

import requests
from dataset import TransformerEmbedding, create_dataloader_v1
import config
from model import GPTModel
from text_generation import generate_text_simple

if __name__ == "__main__":
    text = requests.get(config.URL).text

    print(f"Dataset length (in characters): {len(text):,}")
    print(f'Token number in the text: {len(tiktoken.get_encoding("gpt2").encode(text)):,}')

    embedder = TransformerEmbedding(
        n_vocab=config.VOCAB_SIZE,
        max_length=config.MAX_LENGTH,
        out_dim=config.EMBED_DIM,
    )

    dataloader = create_dataloader_v1(
        txt=text, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    input_ids, target_ids = next(iter(dataloader))
    embeddings = embedder(input_ids)

    print(embeddings.shape)

    model = GPTModel(
        vocab_size=config.VOCAB_SIZE,
        emb_dim=config.EMBED_DIM,
        max_length=config.MAX_LENGTH,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        shortcut_dropout=config.FFN_DROPOUT_RATE,
        mha_dropout=config.MHA_DROPOUT_RATE,
        emb_dropout=config.EMB_DROPOUT_RATE,
        qkv_bias=config.QKV_BIAS
    )


    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

    try:
        import tiktoken
        import torch
        test_seq = "To be, or not to be, that is the question:"

        tokenizer = tiktoken.get_encoding("gpt2")

        encoded = tokenizer.encode(test_seq)
        test_seq = torch.tensor(encoded).unsqueeze(0)
        model.eval()
        text_exp = generate_text_simple(model, idx=test_seq, max_new_tokens=10, context_size=128)

        print("Text generation test passed: ")
        print(tokenizer.decode(text_exp.squeeze().tolist()))

    except ImportError:
        print("tiktoken not installed; skipping text generation test.")