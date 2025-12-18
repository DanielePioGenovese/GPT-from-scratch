import tiktoken
import hydra
from hydra.core.config_store import ConfigStore

import requests
from dataset import create_dataloader_v1
from config import Config
from model import GPTModel
from text_generation import generate_text_simple
from utils import calc_loss_loader

cs = ConfigStore.instance()
cs.store(name='model_config', node=Config)

@hydra.main(version_base=None, config_path=r'D:\Python-Environments\LLM\GPTFrormScratch\src\conf', config_name="config")
def main(cfg: Config):

    text = requests.get(cfg.dataset.url).text

    print(f"Dataset length (in characters): {len(text):,}")
    print(f'Token number in the text: {len(tiktoken.get_encoding("gpt2").encode(text)):,}')

    model = GPTModel(
        vocab_size=cfg.model.vocab_size,
        emb_dim=cfg.model.embed_dim,
        max_length=cfg.model.max_length,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        shortcut_dropout=cfg.model.ffn_dropout_rate,
        mha_dropout=cfg.model.mha_dropout_rate,
        emb_dropout=cfg.model.emb_dropout_rate,
        qkv_bias=cfg.model.qkv_bias
    )


    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

    try:
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=100)
        val_loss = calc_loss_loader(val_dataloader, model, device, num_batches=100)
    print(f"Initial train loss: {train_loss:.4f}")
    print(f"Initial val loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()

