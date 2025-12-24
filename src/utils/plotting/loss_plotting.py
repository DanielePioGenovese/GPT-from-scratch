import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # Sanity check: ensure we have data to plot
    if not train_losses or len(train_losses) == 0:
        print("No loss data to plot.")
        return

    # Ensure data is on CPU and converted to numpy/list for plotting
    if isinstance(epochs_seen, torch.Tensor):
        epochs_seen = epochs_seen.cpu().numpy()
    
    # helper to convert list of tensors to list of floats if needed
    def to_cpu_list(data):
        return [x.item() if isinstance(x, torch.Tensor) else x for x in data]

    train_losses = to_cpu_list(train_losses)
    val_losses = to_cpu_list(val_losses)

    try:
        fig, ax1 = plt.subplots(figsize=(6, 4)) # Slightly larger for better visibility

        # Plot losses
        ax1.plot(epochs_seen, train_losses, label="Training loss", color='blue')
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", color='orange')
        
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        
        # Ensure x-axis doesn't force integers if you have < 1 epoch
        if max(epochs_seen) >= 1:
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Secondary x-axis for Tokens
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0) # Invisible plot to set scale
        ax2.set_xlabel("Tokens seen")
        
        fig.tight_layout()
        # Save before showing, or just save if running on a server
        plt.savefig("loss_plot.png") 
        plt.show()
        
    except Exception as e:
        print(f'Error in plotting the loss: {e}')