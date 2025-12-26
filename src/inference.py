import torch 
import tiktoken

import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path

from model import GPTModel
from conf import Config
from utils import load_checkpoint, generate_and_print_sample

cs = ConfigStore.instance()
cs.store(name='model_inference', node=Config)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def run_inference(cfg: Config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    checkpoint_name = load_checkpoint(
        cfg.model.checkpoint_name,
        cfg.model.checkpoint_path
    )

    if not checkpoint_name:
        print('Error: Checkpoint not found')
        return
    
    full_chk_path = Path(cfg.model.checkpoint_path) / Path(checkpoint_name)

    checkpoint = torch.load(
        full_chk_path,
        map_location=device)
    
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict=state_dict)
    model.to(device)

    tokenizer = tiktoken.get_encoding('gpt2')

    generate_and_print_sample(
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_context=cfg.model.prompt,
        max_new_tokens=cfg.model.max_new_tokens,
        context_size=cfg.model.max_length,
        temperature=cfg.model.temperature,
        top_k=cfg.model.top_k,
        top_p=cfg.model.top_p
    )

if __name__ == '__main__':
    run_inference()

