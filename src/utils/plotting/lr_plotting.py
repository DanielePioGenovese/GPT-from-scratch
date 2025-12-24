import matplotlib.pyplot as plt
import numpy as np


def plot_lr_scheduler(lr_values: list):
    try:
        actual_steps = np.arange(len(lr_values)) 

        plt.figure()
        plt.plot(actual_steps, lr_values, label="Learning Rate")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.title("Learning rate schedule (warmup + cosine decay)")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f'Error in plotting the scheduler: {e}')