import tiktoken
import torch
import hydra
from hydra.core.config_store import ConfigStore
import requests
import torch
from dataset import create_dataloader_v1
from conf import Config
from model import GPTModel
from train import Trainer
from utils import plot_losses

cs = ConfigStore.instance()
cs.store(name="model_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    text = requests.get(cfg.dataset.url).text
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    tokenizer = tiktoken.get_encoding("gpt2")
    trainer = Trainer(model, device)

    print(f"Dataset length (in characters): {len(text):,}")
    print(
        f"Token number in the text: {len(tiktoken.get_encoding('gpt2').encode(text)):,}"
    )

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    split_idx = int(cfg.dataset.train_ratio * len(text))

    train_dataloader = create_dataloader_v1(
        text[:split_idx],
        batch_size=cfg.dataset.batch_size,
        max_length=cfg.model.max_length,
        stride=cfg.dataset.stride,
        drop_last=cfg.dataset.train_drop_last,
        shuffle=cfg.dataset.train_shuffle,
        num_workers=cfg.dataset.num_workers,
    )

    val_dataloader = create_dataloader_v1(
        text[split_idx:],
        batch_size=cfg.dataset.batch_size,
        max_length=cfg.model.max_length,
        stride=cfg.dataset.stride,
        drop_last=cfg.dataset.val_drop_last,
        shuffle=cfg.dataset.val_shuffle,
        num_workers=cfg.dataset.num_workers,
    )

    model.to(device)

    train_losses, val_losses, tokens_seen = trainer.train(
        train_dataloader,
        val_dataloader,
        optimizer="adam",
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        num_epochs=cfg.model.num_epochs,
        eval_freq=500,
        eval_iter=100,
        start_context="Once upon a time",
        tokenizer=tokenizer,
        temperature=cfg.model.temperature,
        top_k=cfg.model.top_k,
        top_p=cfg.model.top_p
    )

    epochs_tensor = torch.linspace(0, cfg.model.num_epochs, steps=len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
