import requests
from dataset import TransformerEmbedding, create_dataloader_v1
import config
from model import GPTModel

if __name__ == "__main__":
    text = requests.get(config.URL).text

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