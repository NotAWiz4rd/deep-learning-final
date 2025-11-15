import torch
from torch.nn import functional as F

from data_loader import DataLoaderLite
from gpt_model import GPT, GPTConfig

# autodetect device
device = "cpu"
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print("using device:", device)

# reproducibility
torch.manual_seed(1337)
if device == "mps":
    torch.mps.manual_seed(1337)
elif device == "cuda":
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=32)

model = GPT(GPTConfig())
model.to(device)

# optimise!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss {loss.item():.5f}")

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
