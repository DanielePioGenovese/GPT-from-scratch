import matplotlib.pyplot as plt
import numpy as np

def plot_lr_scheduler(steps_number, lr_values: list):

    steps = np.arange(0, steps_number)

    plt.figure()
    plt.plot(steps, lr_values, label='Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('LR')
    plt.title('Learning rate schedule (warmup + cosine decay)')
    plt.legend()
    plt.show()