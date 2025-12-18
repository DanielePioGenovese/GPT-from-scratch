import tiktoken
import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
import requests
import torch
from dataset import create_dataloader_v1
from config import Config
from model import GPTModel
from utils import calc_loss_loader, generate_text_simple, calc_loss_batch

def evaluate_model(model, train_loader, val_loader, device, num_batches=None):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50, context_size=128):
    model.eval()
    with torch.no_grad():
        encoded = tokenizer.encode(start_context)
        input_ids = torch.tensor(encoded).unsqueeze(0).to(device)
        generated_ids = generate_text_simple(
            model, input_ids, max_new_tokens, context_size
        )
        generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
        print("Generated text sample:")
        print(generated_text)
    model.train()

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, eval_freq, eval_iter, 
                       start_context, tokenizer):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Epoch {epoch+1}, Step {global_step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

                tqdm.write(
                    f"Step {global_step}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context)
        
    return train_losses, val_losses, track_tokens_seen
cs = ConfigStore.instance()
cs.store(name='model_config', node=Config)

@hydra.main(version_base=None, config_path=r'D:\Python-Environments\LLM\GPTFrormScratch\src\conf', config_name="config")
def main(cfg: Config):

    text = requests.get(cfg.dataset.url).text
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=100)
        val_loss = calc_loss_loader(val_dataloader, model, device, num_batches=100)
    print(f"Initial train loss: {train_loss:.4f}")
    print(f"Initial val loss: {val_loss:.4f}")
    '''

    torch.manual_seed(123)
    model.to(device)
    optmizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_dataloader,
        val_dataloader,
        optmizer,
        device,
        num_epochs=cfg.model.num_epochs,
        eval_freq=500,
        eval_iter=100,
        start_context="Once upon a time",
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    main()

