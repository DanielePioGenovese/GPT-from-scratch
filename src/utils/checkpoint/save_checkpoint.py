import os
import torch

def save_checkpoint(epoch, step, model, optimizer, loss, chk_path, filename=None):
    '''
    This function writes a temporary file to prevent 
    the keyboard from crashing while saving the model.
    '''
    
    if filename is None:
        filename = f"step_{step}.pth"
    
    final_path = chk_path / filename
    temp_path = chk_path / f"{filename}.tmp"
    
    checkpoint = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss if isinstance(loss, float) else loss.item(),
    }
    
    try:
        torch.save(checkpoint, temp_path)
        
        if os.path.exists(final_path):
            os.remove(final_path) 
        os.rename(temp_path, final_path)
        
        print(f"--> Checkpoint salvato correttamente: {final_path}")
    except Exception as e:
        print(f"=== Error saving checkpoint: {e} ===")