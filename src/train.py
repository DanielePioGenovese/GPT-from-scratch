
import requests
from dataset import InputEmbeddings, create_dataloader_v1
import config

if __name__ == "__main__":
    text = requests.get(config.URL).text

    embedder = InputEmbeddings(
        n_vocab=config.VOCAB_SIZE,
        out_dim=config.OUTPUT_DIM
    )

    dataloader = create_dataloader_v1(
        txt=text, 
        batch_size=16
    )

    input_ids, target_ids = next(iter(dataloader))
    embeddings = embedder(input_ids)

    print(embeddings.shape)