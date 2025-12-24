import torch
from torch.amp import GradScaler

from tqdm import tqdm
from pathlib import Path
import sys
import math

# Assuming metrics are in metrics.py or similar
from metrics import calc_loss_batch, calc_loss_loader
from utils import generate_text, get_lr_scheduler, plot_lr_scheduler, load_checkpoint, save_checkpoint


class Trainer:
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
        top_p=None,
    ):
        model.eval()
        encoded = tokenizer.encode(start_context)
        input_ids = torch.tensor(encoded).unsqueeze(0).to(device)

        generated_ids = generate_text(
            model,
            input_ids,
            max_new_tokens,
            context_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
        print(f"\n[---Generated text sample---]:\n{generated_text}\n")
        model.train()

    def _checkpoint(
        self, checkpoint_name, checkpoint_path, steps_per_epoch, chk_path, start_epoch, grad_accumulation_steps
    ):
        check_checkpoint = load_checkpoint(checkpoint_name, checkpoint_path)

        if check_checkpoint == False:
            while True:
                prompt_check = input(
                    "Do you want continue with the training process (yes, no)? "
                ).strip()

                if prompt_check not in ("yes", "no"):
                    print("Type a valid value!")
                elif prompt_check == "yes":
                    global_step = start_epoch * steps_per_epoch - 1
                    break
                else:
                    print("Exiting program.")
                    sys.exit()
            
            global_step, batches_to_skip = 0, 0
            return start_epoch, global_step, batches_to_skip
        
        else:   
            print(f"Resuming from checkpoint: {check_checkpoint}")
            checkpoint = torch.load(
                chk_path / check_checkpoint, map_location=self.device
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] 
            global_step = checkpoint['global_step']
            batches_to_skip = (global_step + 1) * grad_accumulation_steps
            
            return start_epoch, global_step, batches_to_skip

    def _training_loop(
        self,
        lr,
        start_epoch,
        num_epochs,
        train_loader,
        max_steps,
        min_lr,
        warmup_steps,
        scaler,
        val_loader,
        eval_freq,
        eval_iter,
        grad_accumulation_steps, 
        global_step,
        chk_path,
        batches_to_skip
    ):
        train_losses, val_losses, track_tokens_seen, save_lr = [], [], [], []
        token_seen = 0
        max_lr = lr
        
        self.global_step = global_step if global_step >= 0 else 0

        self.optimizer.zero_grad(set_to_none=True) 
        best_loss = torch.inf

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True
            )

            for batch_idx, (input_batch, target_batch) in enumerate(progress_bar):

                if epoch == start_epoch and batch_idx < batches_to_skip:
                    if batch_idx % 100 == 0:
                        progress_bar.set_description(f'Skipping batch {batch_idx}...')
                    continue

                loss = calc_loss_batch(
                    input_batch, target_batch, self.model, self.device
                ) / grad_accumulation_steps 

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    
                    it_for_lr = self.global_step + 1
                    cur_lr = get_lr_scheduler(
                        it_for_lr,
                        max_steps=max_steps,
                        max_lr=max_lr,
                        min_lr=min_lr,
                        warmup_steps=warmup_steps,
                    )
                    save_lr.append(cur_lr)

                    for pg in self.optimizer.param_groups:
                        pg["lr"] = cur_lr

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True) # Reset gradienti
                    
                    self.global_step += 1 

                    if self.global_step % 10 == 0:
                        progress_bar.set_postfix(
                            {"loss": f"{loss.item() * grad_accumulation_steps:.4f}", "lr": f"{cur_lr:.2e}", "step": self.global_step}
                        )

                    if self.global_step % eval_freq == 0:
                        train_loss, val_loss = self._evaluate_model(
                            self.model, train_loader, val_loader, self.device, eval_iter
                        )
                        train_losses.append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
                        val_losses.append(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss)
                        track_tokens_seen.append(token_seen)
                        tqdm.write(f"Step {self.global_step}: Val loss {val_loss:.3f}")
                
                if loss < best_loss:
                    best_loss = loss
                    save_checkpoint(epoch, global_step, self.model, self.optimizer, loss, chk_path)
                
                token_seen += input_batch.numel()

        return save_lr, train_losses, val_losses, track_tokens_seen

    def train(
        self,
        train_loader,
        val_loader,
        optimizer_name,
        lr,
        weight_decay,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
        total_tokens,
        micro_batch_size, 
        seq_len,
        grad_accumulation_steps,
        temperature=1.0,
        top_k=None,
        top_p=None,
        checkpoint_name="last",
        checkpoint_path="checkpoint",
        min_lr: float = 1e-6,
        warmup_steps: int | None = None,
    ):
        # Mixed precision scaler
        scaler = GradScaler()

        # Fix: Use current working directory for checkpoints
        chk_path = Path(checkpoint_path)
        chk_path.mkdir(parents=True, exist_ok=True)

        # Fix: Use AdamW instead of Adam for Transformers
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            # Fallback or other optimizers
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        self.optimizer = optimizer

        num_updates_per_epoch = len(train_loader) // grad_accumulation_steps
        total_steps = num_updates_per_epoch * num_epochs

        tokens_per_update = micro_batch_size * seq_len * grad_accumulation_steps
        raw_updates = (total_tokens * num_epochs) / tokens_per_update

        if warmup_steps is None:
            warmup_steps = max(1, int(0.03 * total_steps))

        # Load Checkpoint if requested
        start_epoch = 0
        global_step = -1

        start_epoch, global_step, batches_to_skip = self._checkpoint(
            checkpoint_name, checkpoint_path, raw_updates, chk_path, start_epoch, grad_accumulation_steps
        )

        save_lr, train_losses, val_losses, track_tokens_seen = self._training_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            
            max_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=lr,
            min_lr=min_lr,
            
            scaler=scaler,
            grad_accumulation_steps=grad_accumulation_steps,
            
            eval_freq=eval_freq,
            eval_iter=eval_iter,
            
            global_step=global_step,
            chk_path=chk_path,
            batches_to_skip=batches_to_skip
        )

        self._generate_and_print_sample(
            self.model,
            tokenizer,
            self.device,
            start_context,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        plot_lr_scheduler(save_lr)

        return train_losses, val_losses, track_tokens_seen
