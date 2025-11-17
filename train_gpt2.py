import math
import os
import time

import torch

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

total_batch_size = 524288  # total tokens per batch
B = 64  # micro batch size
T = 1024
assert total_batch_size % (B * T) == 0, "total batch size must be divisible by B * T"
grad_accumulation_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size} tokens")
print(f"=> calculated gradient accumulation steps: {grad_accumulation_steps}")

train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

if device == "cuda":
    torch.set_float32_matmul_precision('high')  # sets precision down from 'highest', which should use TF32

model = GPT(GPTConfig(vocab_size=50304))  # potentially set vocab size to 50304 to optimise for nice numbers in CUDA
model.to(device)
model = torch.compile(model)

# todo adjust these hyperparameters as for real training run
max_learning_rate = 6e-4 * 2  # double learning rate for better training performance
min_learning_rate = max_learning_rate * 0.1
warmup_steps = 200 # reduced warmup for quicker start; normal GPT would be about 715 steps
max_steps = 19073  # about 1 epoch over the 10B token dataset with our setup


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
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_learning_rate, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if device == "cuda":
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach()

        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        if step > 0 and (step % 3000 == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            # we might also want to add optimizer.state_dict() and
            # rng seeds etc., if we wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accumulation_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accumulation_steps  # scale the loss to account for gradient accumulation
        loss_accum += loss.detach()
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
    tokens_processed = train_loader.B * train_loader.T * grad_accumulation_steps
    tokens_per_sec = tokens_processed / dt
    print(
        f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {learning_rate:.6f} | norm: {norm:.4f} | time: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
