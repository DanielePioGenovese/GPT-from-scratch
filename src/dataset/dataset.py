import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, token_ids, max_length, stride):
        super().__init__()
        self.input_ids = token_ids
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        return (len(self.input_ids) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length

        input_chunk = self.input_ids[start_idx:end_idx]
        target_chunk = self.input_ids[start_idx + 1 : end_idx + 1]

        return input_chunk, target_chunk


def create_dataloader_v1(
    token_ids,
    batch_size,
    max_length=256,
    stride=128,
    shuffle=False,
    drop_last=True,
    num_workers=0,
):
    dataset = GPTDatasetV1(token_ids, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


class InputEmbeddings(torch.nn.Module):
    def __init__(self, n_vocab, out_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_vocab, out_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_length, out_dim):
        super().__init__()
        # GPT impara le posizioni come se fossero parole
        self.pos_embedding = torch.nn.Embedding(max_length, out_dim)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        return self.pos_embedding(positions)


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
