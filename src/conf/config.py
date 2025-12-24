from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str
    checkpoint_path: str
    vocab_size: int
    embed_dim: int
    ff_hidden_dim: int
    num_layers: int
    num_heads: int
    max_length: int          
    output_dim: int
    layer_norm_eps: float
    qkv_bias: bool
    mha_dropout_rate: float
    ffn_dropout_rate: float
    emb_dropout_rate: float
    learning_rate: float    
    min_lr: float            
    weight_decay: float
    warmup_steps: int
    num_epochs: int
    micro_batch_size: int    
    grad_accumulation: int 
    eval_freq: int           
    eval_iter: int           
    temperature: float

    use_checkpoint: Optional[str] = "last"
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    
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
