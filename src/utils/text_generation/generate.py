import torch
import matplotlib.pyplot as plt


def generate_text(
    model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, top_p=None
):
    # Ensure model is in eval mode before entering loop (handled by caller usually, but safe to check)

    for _ in range(max_new_tokens):
        # Crop context if it becomes too long
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # 1. Apply Temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # 2. Apply Top-K
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # 3. Apply Top-P (Nucleus)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to keep at least the first token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        # 4. Final Sampling
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

@torch.no_grad()
def generate_and_print_sample(
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