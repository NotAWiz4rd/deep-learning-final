"""
Generate fantasy location descriptions for finetuning.

- Uses OpenAI's Responses API.
- Saves samples in plain text with <LOC> ... </LOC> separators.
- Prints running and final token counts using GPT-2's tokenizer.

Usage:
    OPENAI_API_KEY=sk-... python generate_locations.py
"""

import time
from pathlib import Path

import tiktoken
from openai import OpenAI

# -----------------------------
# CONFIG
MODEL_NAME = "gpt-4.1-2025-04-14"
NUM_SAMPLES = 200
OUTPUT_FILE = "datasets/finetune_data.txt"
DELAY_BETWEEN_CALLS = 0.1  # seconds; bump up if we hit rate limits

BASE_PROMPT = """Generate a richly detailed description of a fantasy location.
Focus entirely on atmosphere, sensory detail, and environmental storytelling.
Do NOT include meta-notes, game mechanics, NPC stats, encounter summaries, or instructions to the reader.
Write it as if it were a vivid passage from a fantasy novel—immersive, concise, and evocative.

Do not follow a specific structure, but feel free to include any of the following (but do not label the sections) to create a high-quality description:

- Opening anchor – one or two sentences that clearly establish where we are.
- Sensory immersion – describe at least three sensory layers: sight, sound, smell, temperature, textures, movement.
- Environmental details – terrain features, flora, fauna (subtle), architecture, remnants of history, signs of activity or abandonment.
- Atmospheric tone – subtle emotional color (foreboding, tranquil, mysterious, sacred, melancholic, vibrant), but shown through imagery, not stated directly. Do not use mist.
- Closing beat – one line that hints at a deeper story in the location, but without explaining it or turning it into a plot hook.

Length: 150–400 words (choose something within this range!).
Style: elegant, sensory, novelistic.
No lists, no dialogue, no meta text. Just the description.

Be creative!
"""

# OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()

# GPT-2 tokenizer for token counting
enc = tiktoken.get_encoding("gpt2")

out_path = Path(OUTPUT_FILE)
out_path.parent.mkdir(parents=True, exist_ok=True)

total_tokens = 0

# -----------------------------
# MAIN LOOP

for i in range(1, NUM_SAMPLES + 1):
    prompt = BASE_PROMPT

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            temperature=1.1,  # higher creativity
        )
        text = (response.output_text or "").strip()
    except Exception as e:
        print(f"[{i}/{NUM_SAMPLES}] ERROR from API: {e}")
        # If something goes wrong, wait a bit (in case API is overloaded) and skip
        time.sleep(2.0)
        continue

    if not text:
        print(f"[{i}/{NUM_SAMPLES}] Empty response, skipping.")
        continue

    # Count tokens with GPT-2 encoding
    n_tokens = len(enc.encode(text))
    total_tokens += n_tokens

    # Append to file with separators
    with out_path.open("a", encoding="utf-8") as f:
        f.write("<LOC>\n")
        f.write(text.replace("\r\n", "\n").strip())
        f.write("\n</LOC>\n\n")

    print(
        f"[{i}/{NUM_SAMPLES}] "
        f"tokens={n_tokens}, total_tokens={total_tokens}"
    )

    time.sleep(DELAY_BETWEEN_CALLS)

print("\nDone.")
print(f"Samples generated: {NUM_SAMPLES}")
print(f"Approx. total GPT-2 tokens: {total_tokens}")
print(f"Written to: {out_path.resolve()}")
