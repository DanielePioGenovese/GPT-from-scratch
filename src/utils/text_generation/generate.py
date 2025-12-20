import torch

def generate_text(model, idx, max_new_tokens, context_size, temperature=None, top_k=None, top_p=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if temperature is not None and temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        if temperature == 0:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
