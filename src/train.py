
import requests
from dataset import TransformerEmbedding, create_dataloader_v1
import config
from model import MultiHeadAttention

if __name__ == "__main__":
    text = requests.get(config.URL).text

    embedder = TransformerEmbedding(
        n_vocab=config.VOCAB_SIZE,
        max_length=config.MAX_LENGTH,
        out_dim=config.OUTPUT_DIM
    )

    dataloader = create_dataloader_v1(
        txt=text, 
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    input_ids, target_ids = next(iter(dataloader))
    embeddings = embedder(input_ids)

    model = MultiHeadAttention(
        d_in=config.EMBED_DIM,
        d_out=config.OUTPUT_DIM,
        context_length=config.MAX_LENGTH,
        num_heads=2,
        dropout=0.1
    )

    print(embeddings.shape)