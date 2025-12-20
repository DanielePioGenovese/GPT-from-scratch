import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, (
            "Number of tokenized inputs must at least be equal to max_length+1"
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class InputEmbeddings(torch.nn.Module):
    def __init__(self, n_vocab, out_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_vocab, out_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_length, out_dim):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, out_dim, 2) * -(torch.log(torch.tensor(10000.0)) / out_dim)
        )
        pe = torch.zeros(max_length, out_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, input_ids):
        sequence_length = input_ids.size(1)
        return self.pe[:, :sequence_length]


class TransformerEmbedding(torch.nn.Module):
    def __init__(self, n_vocab, max_length, out_dim, emb_dropout=0.1):
        super().__init__()
        self.token_embedding = InputEmbeddings(n_vocab, out_dim)
        self.position_embeddings = PositionalEmbedding(max_length, out_dim)
        self.dropout = torch.nn.Dropout(emb_dropout)

    def forward(self, input_ids):
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embeddings(input_ids)
        combined_embeds = token_embeds + pos_embeds

        return self.dropout(combined_embeds)


def create_dataloader_v1(
    txt,
    batch_size,
    max_length=256,
    stride=128,
    shuffle=False,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
