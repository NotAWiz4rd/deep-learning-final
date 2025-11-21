import tiktoken
import torch

from gpt_model import GPT

# ======================
# CONFIG
# ======================

CKPT_PATH = "finetune_runs/final_lr_0.0002/finetune_epoch4.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TOKENS = 300  # ~1–2 paragraphs
TEMPERATURE = 0.9
TOP_K = 50

PROMPT = (  # amy further prompt confuses the model as it is out of distribution to its training data
    "\n"
)


# ======================
# MODEL LOADING
# ======================

def load_finetuned_model(ckpt_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint does not contain a 'model' key.")

    state = ckpt["model"]
    config = ckpt["config"]

    model = GPT(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model


# ======================
# SAMPLING FUNCTION
# ======================

def sample(model: GPT, enc, prompt: str, max_new_tokens=200,
           temperature=1.0, top_k=None, device="cpu"):
    model.eval()

    # Encode prompt
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long)[None].to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                min_top_val = top_vals[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_top_val, torch.full_like(logits, -1e10), logits)

            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return enc.decode(input_ids[0].tolist())


# ======================
# MAIN
# ======================

def main():
    print("Loading checkpoint:", CKPT_PATH)
    model = load_finetuned_model(CKPT_PATH, DEVICE)
    enc = tiktoken.get_encoding("gpt2")

    print("\n=== Generated Samples ===\n")

    for i in range(3):
        text = sample(
            model=model,
            enc=enc,
            prompt=PROMPT,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            device=DEVICE,
        )
        print(f"\n--- SAMPLE {i + 1} ---\n{text}\n")


if __name__ == "__main__":
    main()
