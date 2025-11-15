import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B  # batch size
        self.T = T  # sequence length
        self.tokenizer = tiktoken.get_encoding("gpt2")  # Using GPT-2 tokenizer

        # at init load tokens from disk and store them in memory
        with open('datasets/tiny-shakespeare.txt', 'r') as f:
            text = f.read()

        tokens = self.tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches of size {B}, sequence length {T}")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buffer[:-1].view(B, T)  # inputs
        y = buffer[1:].view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would go out of bounds, reset position to zero
        if self.current_position + B * T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y
