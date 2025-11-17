# GPT-2 From Scratch: Deep Learning Final Project

Final project for the CU Boulder MSCS Deep Learning course. This project involves building a GPT-2-Small-like transformer model from scratch, training it on large-scale web data, and then fine-tuning it for a specific use case to achieve performance close to larger models in that domain.

## Project Goals

1. **Rebuild GPT-2-Small**: Implement a transformer architecture similar to GPT-2-Small (124M parameters) from the ground up
2. **Large-Scale Pretraining**: Train the model on the 10B token FineWeb-Edu dataset
3. **Domain-Specific Fine-tuning**: Fine-tune the pretrained model for a specific use case (currently considering RPG dungeon descriptions) to demonstrate that smaller, specialized models can approach the performance of larger general-purpose models

## Setup

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:
- PyTorch 2.9.1+
- tiktoken (GPT-2 tokenizer)
- datasets (HuggingFace)
- tqdm

### Hardware Recommendations

- **For full training**: CUDA-capable GPU with at least 16GB VRAM recommended
- **For testing/smaller runs**: CPU or MPS (Apple Silicon) will work but will be significantly slower

## Usage

### Step 1: Download and Prepare the Dataset

First, download, tokenize, and shard the FineWeb-Edu dataset (10B token sample):

```bash
python fineweb_edu.py
```

This script will:
- Download the FineWeb-Edu 10B token sample from HuggingFace
- Tokenize the text using the GPT-2 tokenizer
- Split the data into shards of 100M tokens each
- Save the sharded data to `datasets/edu_fineweb10B/`
- Automatically create a validation split (first shard) and training split (remaining shards)

**Note**: This process is computationally intensive and will take some time depending on your CPU cores.

### Step 2: Train the Model

Run the training script to train GPT-2-Small on the prepared dataset:

```bash
python train_gpt2.py
```

The training script:
- Trains a GPT-2-Small model (124M parameters) with vocabulary size 50304
- Uses a batch size of 524,288 tokens with gradient accumulation
- Implements cosine learning rate decay with warmup
- Runs for ~19,073 steps (approximately 1 epoch over 10B tokens)
- Saves checkpoints every 5,000 steps to the `log/` directory
- Evaluates validation loss every 100 steps

#### Adjusting Training Parameters

For smaller/faster training runs, modify these parameters in `train_gpt2.py`:

- `B` (batch size): Reduce from 64 for less memory usage
- `T` (sequence length): Reduce from 1024 for faster training
- `max_steps`: Reduce for shorter training duration
- `max_learning_rate`: Adjust for different convergence behavior

Training logs and checkpoints are saved to the `log/` directory.

## Project Structure

- `fineweb_edu.py` - Dataset download, tokenization, and sharding
- `train_gpt2.py` - Main training loop with learning rate scheduling
- `gpt_model.py` - GPT model implementation
- `transformer_modules.py` - Core transformer building blocks
- `data_loader.py` - Data loading utilities for sharded data
- `requirements.txt` - Python dependencies

## Next Steps

- Complete pretraining on the full 10B token dataset
- Select and prepare a domain-specific dataset for fine-tuning (e.g., RPG dungeon descriptions)
- Implement fine-tuning script and evaluate performance against larger general-purpose models
- Compare specialized model performance to GPT-3/GPT-4 baselines on the target domain

## License

Academic project for CU Boulder MSCS Deep Learning course.
