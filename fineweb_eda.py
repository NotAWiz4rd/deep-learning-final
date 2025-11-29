import os
import re
import numpy as np
import json
from collections import Counter

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from wordcloud import WordCloud
except ImportError:
    print("wordcloud not found. Install with: pip install wordcloud")
    raise


# ============================================================
# Configuration
# ============================================================

MAX_DOCS = 150_000     # number of documents to sample
MAX_TOKEN_HIST = 3000
MAX_CHAR_HIST = 15_000
TOP_K_WORDS = 30

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] EDA outputs will be saved to: {OUTPUT_DIR}/")


# ============================================================
# Load dataset (streaming)
# ============================================================

print("[INFO] Loading dataset (streaming mode)...")
ds = load_dataset("HuggingFaceFW/FineWeb-Edu", "sample-10BT", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# ============================================================
# Collect statistics
# ============================================================

lengths_tokens = []
lengths_chars = []
word_freq = Counter()

print(f"[INFO] Sampling up to {MAX_DOCS} documents...")

for i, row in enumerate(ds):
    text = row["text"]
    if not text:
        continue

    lengths_chars.append(len(text))

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    lengths_tokens.append(len(tokens))

    # Simple word-level frequency
    words = re.findall(r"\b\w+\b", text.lower())
    word_freq.update(words)

    if (i + 1) % 5000 == 0:
        print(f"  processed {i+1} documents...")

    if i + 1 >= MAX_DOCS:
        break

print("[INFO] Sampling done.")
print(f"[INFO] Collected {len(lengths_tokens)} valid documents.")


# ============================================================
# Save raw statistics
# ============================================================

np.save(os.path.join(OUTPUT_DIR, "lengths_tokens.npy"), np.array(lengths_tokens))
np.save(os.path.join(OUTPUT_DIR, "lengths_chars.npy"), np.array(lengths_chars))

with open(os.path.join(OUTPUT_DIR, "word_frequencies.json"), "w") as f:
    json.dump(word_freq.most_common(10_000), f)

print("[INFO] Raw stats saved.")


# ============================================================
# Plot 1: Token length histogram
# ============================================================

print("[INFO] Saving token histogram...")
clipped_tokens = np.clip(lengths_tokens, 0, MAX_TOKEN_HIST)

plt.figure(figsize=(8, 5))
plt.hist(clipped_tokens, bins=60)
plt.title("Document Length Distribution (tokens)")
plt.xlabel(f"Tokens per document (clipped at {MAX_TOKEN_HIST})")
plt.ylabel("Count")
plt.yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "token_length_hist.png"), dpi=200)
plt.close()


# ============================================================
# Plot 2: Character length histogram
# ============================================================

print("[INFO] Saving character histogram...")
clipped_chars = np.clip(lengths_chars, 0, MAX_CHAR_HIST)

plt.figure(figsize=(8, 5))
plt.hist(clipped_chars, bins=60)
plt.title("Document Length Distribution (characters)")
plt.xlabel(f"Characters per document (clipped at {MAX_CHAR_HIST})")
plt.ylabel("Count")
plt.yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "char_length_hist.png"), dpi=200)
plt.close()


# ============================================================
# Plot 3: Top words bar chart
# ============================================================

print("[INFO] Saving top-words bar chart...")
stopwords = {
    "the","of","and","to","in","a","is","for","on","that","with",
    "as","by","it","at","from","this","an","be","or","are","was",
    "s", "they", "their", "you", "which", "your", "we", "were", "these",
    "have", "has", "its", "not", "t", "1", "2", "one", "two", "he", "she",
    "being", "them", "her", "him", "so"
}

filtered = Counter({w: c for w, c in word_freq.items() if w not in stopwords})
most_common = filtered.most_common(TOP_K_WORDS)

labels = [w for w, c in most_common]
counts = [c for w, c in most_common]

plt.figure(figsize=(10, 5))
plt.bar(range(len(labels)), counts)
plt.xticks(range(len(labels)), labels, rotation=60, ha='right')
plt.title(f"Top {TOP_K_WORDS} Most Frequent Words (stopwords removed)")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_words_bar_chart.png"), dpi=200)
plt.close()


# ============================================================
# Plot 4: Wordcloud
# ============================================================

print("[INFO] Saving wordcloud...")
wc = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    max_words=200,
).generate_from_frequencies(filtered)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of Frequent Words (FineWeb-Edu sample)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"), dpi=200)
plt.close()


# ============================================================
# Summary printout
# ============================================================

print("\n=== SUMMARY ===")
print(f"Documents sampled: {len(lengths_tokens)}")
print(f"Median token length: {np.median(lengths_tokens):.2f}")
print(f"95th percentile token length: {np.percentile(lengths_tokens, 95):.2f}")
print(f"Max token length (clipped): {max(lengths_tokens)}")

print("\nSaved files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", fname)

print("\n[INFO] EDA complete.")