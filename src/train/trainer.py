import torch
from tqdm import tqdm
from metrics import calc_loss_batch, calc_loss_loader
from utils import generate_text
from pathlib import Path

class Trainer:
    """Trainer class to handle training and evaluation of the model."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def _evaluate_model(
        self, model, train_loader, val_loader, device, num_batches=None
    ):
        model.eval()
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
        model.train()
        return train_loss, val_loss

    @torch.no_grad()
    def _generate_and_print_sample(
        self,
        model,
        tokenizer,
        device,
        start_context,
        max_new_tokens=50,
        context_size=128,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        model.eval()
        encoded = tokenizer.encode(start_context)
        input_ids = torch.tensor(encoded).unsqueeze(0).to(device)
        generated_ids = generate_text(
            model, input_ids, max_new_tokens, context_size, temperature=temperature, top_k=top_k, top_p=top_p
        )
        generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
        print("Generated text sample:")
        print(generated_text)
        model.train()

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        lr,
        weight_decay,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
        temperature=1.0,
        top_k=None,
        top_p=None,
        use_checkpoint:str=None, 
        checkpoint_path='checkpoint'
    ):

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent

        chk_path = project_root / checkpoint_path
        chk_path.mkdir(exist_ok=True)

        if optimizer != "adam":
            raise ValueError("Currently only 'adam' optimizer is supported.")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if use_checkpoint != None:
            self.model.load_state_dict(torch.load(chk_path / use_checkpoint)['model_state_dict'])

        train_losses, val_losses, track_tokens_seen = [], [], []
        token_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True
            )
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(
                    input_batch, target_batch, self.model, self.device
                )
                loss.backward()
                optimizer.step()
                token_seen += input_batch.numel()
                global_step += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(
                        self.model, train_loader, val_loader, self.device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(token_seen)
                    print(
                        f"Epoch {epoch + 1}, Step {global_step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                    )

                    tqdm.write(
                        f"Step {global_step}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

            self._generate_and_print_sample(
                self.model, tokenizer, self.device, start_context, temperature=temperature, top_k=top_k, top_p=top_p
            )

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss.item()
            }

            #Checkpoint
            torch.save(checkpoint, chk_path / f'{epoch}.pth')


        return train_losses, val_losses, track_tokens_seen
