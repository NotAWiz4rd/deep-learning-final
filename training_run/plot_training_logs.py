#!/usr/bin/env python
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_log(path: str):
    text = Path(path).read_text()

    # Training lines:
    # step 56500 | loss: 2.963373 | lr: 0.000120 | norm: 0.4342 | time: 2366.80ms | tok/sec: 221517.41
    train_re = re.compile(
        r"step\s+(\d+)\s*\|\s*loss:\s*([0-9.]+)\s*\|\s*lr:\s*([0-9.]+)\s*"
        r"\|\s*norm:\s*([0-9.]+)\s*\|\s*time:\s*([0-9.]+)ms\s*\|\s*tok/sec:\s*([0-9.]+)"
    )

    # Validation lines:
    # Validation lines look like: "56750 val 59.9773" (sometimes stuck to previous float)
    val_re = re.compile(r"(\d+)\s+val\s+([0-9.]+)")

    train_rows = []
    for m in train_re.finditer(text):
        step = int(m.group(1))
        loss = float(m.group(2))
        lr = float(m.group(3))
        norm = float(m.group(4))
        time_ms = float(m.group(5))
        tok_sec = float(m.group(6))
        train_rows.append(
            {
                "step": step,
                "loss": loss,
                "lr": lr,
                "norm": norm,
                "time_ms": time_ms,
                "tok_sec": tok_sec,
            }
        )

    val_rows = []
    for m in val_re.finditer(text):
        step = int(m.group(1))
        val_loss = float(m.group(2))
        val_rows.append({"step": step, "val_loss": val_loss})

    if not train_rows:
        raise RuntimeError("No training lines matched. Check regex/log format.")

    df_train = pd.DataFrame(train_rows).sort_values("step").reset_index(drop=True)
    df_val = None
    if val_rows:
        df_val = pd.DataFrame(val_rows).sort_values("step").reset_index(drop=True)

    return df_train, df_val


def plot_series(x, y, xlabel, ylabel, title, out_path, y2=None, y2_label=None):
    plt.figure()
    plt.plot(x, y, label=ylabel)
    if y2 is not None:
        plt.plot(x, y2, linestyle="--", label=y2_label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if y2 is None else "")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_logs.py path/to/log.txt")
        sys.exit(1)

    log_path = sys.argv[1]
    out_dir = Path(log_path).with_suffix("")  # e.g. log.txt -> log/
    out_dir.mkdir(exist_ok=True)

    df_train, df_val = parse_log(log_path)

    # Smoothed loss for nicer curve
    df_train["loss_smooth"] = df_train["loss"].rolling(window=50, min_periods=1).mean()

    # 1) Training loss
    plot_series(
        df_train["step"],
        df_train["loss"],
        xlabel="Step",
        ylabel="Training loss",
        title="Training Loss",
        out_path=out_dir / "loss.png",
    )

    # 1b) Training loss with smoothing
    plot_series(
        df_train["step"],
        df_train["loss"],
        xlabel="Step",
        ylabel="Training loss",
        title="Training Loss (with Smoothed Curve)",
        out_path=out_dir / "loss_with_smooth.png",
        y2=df_train["loss_smooth"],
        y2_label="Smoothed loss (window=50)",
    )

    # 2) Learning rate
    plot_series(
        df_train["step"],
        df_train["lr"],
        xlabel="Step",
        ylabel="Learning rate",
        title="Learning Rate Schedule",
        out_path=out_dir / "lr.png",
    )

    # 3) Gradient norm
    plot_series(
        df_train["step"],
        df_train["norm"],
        xlabel="Step",
        ylabel="Gradient norm",
        title="Gradient Norm Over Training",
        out_path=out_dir / "norm.png",
    )

    # 4) Tokens per second
    plot_series(
        df_train["step"],
        df_train["tok_sec"],
        xlabel="Step",
        ylabel="Tokens / second",
        title="Throughput (tokens/sec)",
        out_path=out_dir / "tok_per_sec.png",
    )

    # 5) Step time
    plot_series(
        df_train["step"],
        df_train["time_ms"],
        xlabel="Step",
        ylabel="Time per step (ms)",
        title="Step Time",
        out_path=out_dir / "time_ms.png",
    )

    # 6) Validation loss: separate plot
    if df_val is not None and len(df_val) > 0:
        plt.figure()
        plt.plot(df_val["step"], df_val["val_loss"], marker="o")
        plt.xlabel("Step")
        plt.ylabel("Validation loss")
        plt.title("Validation Loss vs Step")
        plt.tight_layout()
        plt.savefig(out_dir / "val_loss.png", dpi=200)
        plt.close()

        # 6b) Combined figure: smoothed train loss vs val loss on twin axes
        fig, ax1 = plt.subplots()
        ax1.plot(df_train["step"], df_train["loss_smooth"], label="Train loss (smooth)")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Train loss", color="black")

        ax2 = ax1.twinx()
        ax2.plot(
            df_val["step"],
            df_val["val_loss"],
            "o-",
            label="Validation loss",
        )
        ax2.set_ylabel("Validation loss", color="black")

        fig.suptitle("Train vs Validation Loss")
        fig.tight_layout()
        fig.savefig(out_dir / "loss_train_vs_val.png", dpi=200)
        plt.close(fig)

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()