from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_heads: int
    num_layers: int
    qkv_bias: bool
    mha_dropout_rate: float
    ffn_dropout_rate: float
    emb_dropout_rate: float
    layer_norm_eps: float
    embed_dim: int
    ff_hidden_dim: int
    output_dim: int
    vocab_size: int
    max_length: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    eval_freq: int
    eval_iter: int
    temperature: float


@dataclass
class DataConfig:
    url: str
    data_dir: str
    batch_size: int
    stride: int
    train_ratio: float
    train_shuffle: bool
    val_shuffle: bool
    train_drop_last: bool
    val_drop_last: bool
    num_workers: int


@dataclass
class Config:
    model: ModelConfig
    dataset: DataConfig
