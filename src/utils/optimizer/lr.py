import math

# Function to get learning rate scheduler

def get_lr_scheduler(iter: int, max_steps : int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    ''' Returns the learning rate at a given iteration using a linear warmup and cosine decay schedule. '''

    if iter < warmup_steps:
        return max_lr * (iter + 1) / (warmup_steps + 1)

    if iter >= max_steps:
        return min_lr
    
    decay_ratio = (iter - warmup_steps) / max(1, (max_steps - warmup_steps))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


