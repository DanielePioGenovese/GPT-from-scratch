import tiktoken
import torch
import hydra
import requests
from hydra.core.config_store import ConfigStore

from dataset import create_dataloader_v1

from conf import Config
from model import GPTModel
from train import Trainer
from utils import plot_losses, prepare_data
import numpy as np
import os

cs = ConfigStore.instance()
cs.store(name="model_config", node=Config)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):

    if not os.path.exists('dataset_train.bin'):
        prepare_data()

    data_map = np.memmap('dataset_train.bin', dtype=np.uint16, mode='r')
    token_ids_tensor = torch.from_numpy(data_map.astype(np.int64))
    total_tokens = len(token_ids_tensor)
    
    print(f"Dataset on the disk, total tokens: {total_tokens:,}")

    split_idx = int(cfg.dataset.train_ratio * total_tokens)
    train_data = token_ids_tensor[:split_idx]
    val_data = token_ids_tensor[split_idx:]

    train_dataloader = create_dataloader_v1(
        train_data,
        batch_size=cfg.model.micro_batch_size,
        max_length=cfg.model.max_length,
        stride=cfg.model.max_length, 
        shuffle=cfg.dataset.train_shuffle,
        num_workers=cfg.dataset.num_workers,
    )

    val_dataloader = create_dataloader_v1(
        val_data,
        batch_size=cfg.model.micro_batch_size,
        max_length=cfg.model.max_length,
        stride=cfg.model.max_length,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(
        vocab_size=cfg.model.vocab_size,
        emb_dim=cfg.model.embed_dim,
        max_length=cfg.model.max_length,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        shortcut_dropout=cfg.model.ffn_dropout_rate,
        mha_dropout=cfg.model.mha_dropout_rate,
        emb_dropout=cfg.model.emb_dropout_rate,
        qkv_bias=cfg.model.qkv_bias,
    )


    number_of_paramers = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f'Number of model parameters :{number_of_paramers:,}')
    
    model.to(device)

    trainer = Trainer(model, device)

    print("\n")
    print("===" * 5)
    print("Model: ", cfg.model.model_name)
    print("===" * 5)
    print("\n")

    train_losses, val_losses, tokens_seen = trainer.train(
        # --- Data & Tokenizer ---
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        tokenizer=tokenizer,
        total_tokens=total_tokens,             
        seq_len=cfg.model.max_length,         
        
        # --- Optimization & Scaling ---
        optimizer_name="AdamW",
        lr=cfg.model.learning_rate,
        min_lr=cfg.model.min_lr,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        num_epochs=cfg.model.num_epochs,
        micro_batch_size=cfg.model.micro_batch_size,
        grad_accumulation_steps=cfg.model.grad_accumulation,
        
        # --- Monitoring & Eval ---
        eval_freq=cfg.model.eval_freq,
        eval_iter=cfg.model.eval_iter,
        
        # --- Text Generation (Testing) ---
        start_context="Once upon a time",
        temperature=cfg.model.temperature,
        top_k=cfg.model.top_k,
        top_p=cfg.model.top_p,
        
        # --- Checkpointing ---
        checkpoint_name=cfg.model.use_checkpoint,
        checkpoint_path=cfg.model.checkpoint_path
    )

    # Plotting
    epochs_tensor = torch.linspace(0, cfg.model.num_epochs, steps=len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
