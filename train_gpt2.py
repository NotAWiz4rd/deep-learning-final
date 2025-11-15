import math
import time

import torch
from torch.nn import functional as F

from data_loader import DataLoaderLite
from gpt_model import GPT, GPTConfig

# autodetect device
device = "cpu"
if torch.backends.mps.is_available():
    device = "cpu"  # default to cpu on Mac for now, as mps is super slow
elif torch.cuda.is_available():
    device = "cuda"
print("using device:", device)

# reproducibility
torch.manual_seed(1337)
if device == "mps":
    torch.mps.manual_seed(1337)
elif device == "cuda":
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=8, T=1024)

model = GPT(GPTConfig())  # todo potentially set vocab size to 50304 to optimise for nice numbers in CUDA
model.to(device)
model = torch.compile(model)

max_learning_rate = 6e-4
min_learning_rate = max_learning_rate * 0.1
warmup_steps = 4
max_steps = 20


def get_learning_rate(step):
    # 1) linear warmup for warmup_step steps
    if step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps

    # 2) if step > max_steps: return min_learning_rate
    if step > max_steps:
        return min_learning_rate

    # 3) in between, use cosine decay down to min_learning_rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


# optimise!
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-9)
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    if device == "cuda":
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
    else:
        logits, loss = model(x, y)
    loss.backward()
    # gradient clipping to prevent shocking model updates
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    learning_rate = get_learning_rate(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    optimizer.step()

    # wait to finish work before measuring time
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elif device == "cpu":
        torch.cpu.synchronize()
    t1 = time.time()
    dt = (t1 - t0)  # time difference
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    print(
        f"step {step:4d} | loss: {loss.item():.6f} | time: {dt * 1000:.2f}ms | norm: {norm:.4f} | speed: {tokens_per_sec:.2f} tokens/sec")

import sys;

sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)  # (5, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (5, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)  # (5, vocab_size)
        # do top-k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1)  # (5, 1)
        # gather the corresponding token indices
        next_token = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, next_token), dim=1)  # (5, T+1)

# decode and print the sequences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

print("didn't crash yay!")
