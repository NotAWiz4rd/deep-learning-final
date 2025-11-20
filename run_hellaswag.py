import torch
from torch.nn import functional as F

from gpt_model import GPT, GPTConfig
from hellaswag import iterate_examples, render_example


# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "training_run/log/model_19072.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

print(ckpt.keys())  # should show: dict_keys(['model', 'config', 'step', 'val_loss'])

raw_state = ckpt["model"]

# Strip the `_orig_mod.` prefix that comes from compiled/wrapped models
fixed_state = {}
prefix = "_orig_mod."
for k, v in raw_state.items():
    if k.startswith(prefix):
        new_k = k[len(prefix):]
    else:
        new_k = k
    fixed_state[new_k] = v

# 1. Rebuild the architecture
if "config" in ckpt and isinstance(ckpt["config"], GPTConfig):
    config = ckpt["config"]
else:
    # or recreate manually if needed
    config = GPTConfig(vocab_size=50304)

model = GPT(config)
missing, unexpected = model.load_state_dict(fixed_state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.to(device)
model.eval()

num_correct_norm = 0
num_total = 0
for i, example in enumerate(iterate_examples("val")):
    # render the example into tokens and labels
    _, tokens, mask, label = render_example(example)
    tokens = tokens.to(device)
    mask = mask.to(device)
    # get the logits
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
    num_total += 1
    num_correct_norm += int(pred_norm == label)
    print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")

acc_norm = num_correct_norm / num_total
print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
