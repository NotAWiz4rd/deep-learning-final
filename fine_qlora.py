import math
import os
from pathlib import Path
from typing import List, Tuple

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

from gpt_model import GPT, GPTConfig

# =========================
# CONFIG
# =========================

CKPT_PATH = "training_run/log/model_19072.pt"
DATA_FILE = "datasets/finetune_data.txt"
OUT_DIR = "finetune_runs"

BLOCK_SIZE = 512  # sequence length for finetuning
BATCH_SIZE = 8  # per-step batch size
GRAD_ACCUM_STEPS = 8  # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_WORKERS = 2

SEARCH_EPOCHS = 1  # short runs for LR search
FINAL_EPOCHS = 7  # longer run with best LR

LEARNING_RATES = [1e-4, 2e-4, 5e-4, 1e-3, 4e-3, 7e-3]  # LR sweep

WEIGHT_DECAY = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# MODEL LOADING
# =========================

def load_base_model(ckpt_path: str, device: torch.device) -> GPT:
    """Load GPT from compiled checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} doesn't look like expected dict with 'model' key.")

    raw_state = ckpt["model"]

    # Strip _orig_mod. prefix
    fixed_state = {}
    prefix = "_orig_mod."
    for k, v in raw_state.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        fixed_state[new_k] = v

    # Config: use stored config if it's GPTConfig, else recreate
    if "config" in ckpt and isinstance(ckpt["config"], GPTConfig):
        config = ckpt["config"]
    else:
        config = GPTConfig(vocab_size=50304)

    model = GPT(config)
    missing, unexpected = model.load_state_dict(fixed_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(device)
    model.train()
    return model


# =========================
# DATASET
# =========================

class PackedTextDataset(Dataset):
    """Pack all tokens into one stream and chunk into (input, target) sequences."""

    def __init__(self, tokens: List[int], block_size: int):
        self.block_size = block_size
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # number of full (block_size + 1) chunks we can extract
        self.num_chunks = (len(self.tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size + 1  # +1 for next-token target
        chunk = self.tokens[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_loc_blocks(path: str) -> List[str]:
    """Parse <LOC> ... </LOC> blocks from file."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = []
    current = []
    in_block = False

    for line in text.splitlines():
        s = line.strip()
        if s == "<LOC>":
            in_block = True
            current = []
            continue
        elif s == "</LOC>":
            in_block = False
            sample = "\n".join(current).strip()
            if sample:
                blocks.append(sample)
            current = []
            continue

        if in_block:
            current.append(line.rstrip("\n"))

    return blocks


def build_datasets(data_file: str, block_size: int, val_frac: float = 0.1):
    """Tokenize <LOC> blocks with GPT-2 BPE, pack, and return train/val datasets."""
    enc = tiktoken.get_encoding("gpt2")

    locs = load_loc_blocks(data_file)
    if not locs:
        raise ValueError(f"No <LOC> blocks found in {data_file}")
    print(f"Loaded {len(locs)} location samples from {data_file}")

    # Encode and pack into one long stream, separating locations with an EOS token
    all_ids = []
    eos_id = enc.eot_token  # 50256
    for i, loc in enumerate(locs):
        ids = enc.encode(loc)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    print(f"Total token count (approx GPT-2 BPE): {len(all_ids)}")

    dataset = PackedTextDataset(all_ids, block_size)

    # Train/val split
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    print(f"Created {n_train} train sequences and {n_val} val sequences.")
    return train_ds, val_ds


def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y


# =========================
# TRAINING
# =========================

def run_epoch(
        model: GPT,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler | None,
        train: bool,
        desc: str = "",
) -> float:
    """Run one epoch; if train=True, do updates, else just eval. Returns average loss."""
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    progress = tqdm(loader, desc=desc, leave=False)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        # show running loss in the bar
        progress.set_postfix(loss=f"{loss_val:.4f}")

    return float(sum(losses) / len(losses))


def configure_optimizer(model: GPT, lr: float, device: torch.device) -> torch.optim.Optimizer:
    """Use built-in optimizer config."""
    return model.configure_optimizers(weight_decay=WEIGHT_DECAY, learning_rate=lr, device=device.type)


def train_with_lr(
        lr: float,
        train_ds: Dataset,
        val_ds: Dataset,
        device: torch.device,
        num_epochs: int,
        description: str = "",
) -> float:
    """Train a fresh model from checkpoint with given LR, return best val loss."""
    print(f"\n=== Training with LR={lr} {description} ===")
    model = load_base_model(CKPT_PATH, device)
    optimizer = configure_optimizer(model, lr, device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=collate_batch,
    )

    best_val = math.inf
    for epoch in range(1, num_epochs + 1):
        train_desc = f"LR={lr} [train] epoch {epoch}/{num_epochs}"
        val_desc = f"LR={lr} [val]   epoch {epoch}/{num_epochs}"

        train_loss = run_epoch(model, train_loader, optimizer, device, scaler, train=True, desc=train_desc)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=device, scaler=None, train=False, desc=val_desc)

        print(f"[LR={lr}] epoch {epoch}/{num_epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        best_val = min(best_val, val_loss)

    return best_val


def final_train_and_save(
        lr: float,
        train_ds: Dataset,
        val_ds: Dataset,
        device: torch.device,
        num_epochs: int,
        out_dir: str,
):
    """Final training with best LR, then save checkpoint."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== FINAL TRAIN with LR={lr}, epochs={num_epochs} ===")
    model = load_base_model(CKPT_PATH, device)
    optimizer = configure_optimizer(model, lr, device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=collate_batch,
    )

    best_val = math.inf
    for epoch in range(1, num_epochs + 1):
        train_desc = f"[FINAL] LR={lr} [train] epoch {epoch}/{num_epochs}"
        val_desc = f"[FINAL] LR={lr} [val]   epoch {epoch}/{num_epochs}"

        train_loss = run_epoch(model, train_loader, optimizer, device, scaler, train=True, desc=train_desc)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=device, scaler=None, train=False, desc=val_desc)

        print(f"[FINAL] epoch {epoch}/{num_epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        best_val = min(best_val, val_loss)

        # save an epoch-level checkpoint
        ckpt = {
            "model": model.state_dict(),
            "config": model.config,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(out_dir, f"finetune_epoch{epoch}.pt"))

    # Save final checkpoint + best val in a separate file
    final_ckpt = {
        "model": model.state_dict(),
        "config": model.config,
        "epochs": num_epochs,
        "best_val_loss": best_val,
        "lr": lr,
    }
    torch.save(final_ckpt, os.path.join(out_dir, "finetune_final.pt"))
    print(f"Saved final finetuned model to {out_dir}/finetune_final.pt with best_val_loss={best_val:.4f}")


# =========================
# MAIN
# =========================

def main():
    print("Using device:", DEVICE)
    if DEVICE.type != "cuda":
        print("WARNING: CUDA not available, this will be slow. Intended for GPU.")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build datasets
    train_ds, val_ds = build_datasets(DATA_FILE, BLOCK_SIZE)

    # Hyperparameter search over learning rates
    results = []
    for lr in LEARNING_RATES:
       val_loss = train_with_lr(
           lr=lr,
           train_ds=train_ds,
           val_ds=val_ds,
           device=DEVICE,
           num_epochs=SEARCH_EPOCHS,
           description="(LR sweep)",
       )
       results.append((lr, val_loss))

    print("\n=== LR SEARCH RESULTS ===")
    for lr, loss in results:
       print(f"LR={lr}: best_val_loss={loss:.4f}")

    # Choose best LR
    best_lr, best_loss = min(results, key=lambda x: x[1])
    print(f"\nBest LR: {best_lr} (val_loss={best_loss:.4f})")

    best_lr = 2e-4

    # Final training
    final_out_dir = os.path.join(OUT_DIR, f"final_lr_{best_lr:g}")
    final_train_and_save(
        lr=best_lr,
        train_ds=train_ds,
        val_ds=val_ds,
        device=DEVICE,
        num_epochs=FINAL_EPOCHS,
        out_dir=final_out_dir,
    )


if __name__ == "__main__":
    main()
