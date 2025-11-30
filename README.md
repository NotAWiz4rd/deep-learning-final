# Generating Fantasy Location Descriptions with a 124M LLM

Final project for the CU Boulder MSCS Deep Learning course by Max Werner.

## Project Overview

This project explores a fundamental question in modern AI: **Can a small, fully self-trained language model generate high-quality domain-specific text without requiring massive computational resources or proprietary systems?**

Large modern language models are computationally expensive, difficult to deploy on edge devices, and entirely inaccessible for real-time use in settings like video games or procedural RPG tools. This project demonstrates that a 124M-parameter GPT-2-style model, trained from scratch and fine-tuned on a specialized corpus, can generate rich, evocative fantasy location descriptions suitable for games and narrative engines—all while remaining small enough to run on consumer hardware.

The project implements the full deep learning lifecycle:
- **Architecture implementation**: GPT-2-Small (124M parameters) built from the ground up
- **Large-scale pretraining**: Training on 10 billion tokens of educational web text (FineWeb-Edu)
- **Benchmark evaluation**: Testing general linguistic ability on HellaSwag commonsense reasoning
- **Domain specialization**: Fine-tuning on 1,000 curated fantasy descriptions to achieve stylistic mastery

**Key Finding**: The pretrained model achieved 32.5% on HellaSwag (exceeding GPT-2 Small's 28.9% baseline), and the fine-tuned model generates coherent, atmospheric fantasy descriptions with rich sensory details—demonstrating that smaller, specialized models are a viable path for resource-efficient, domain-specific text generation.

## Project Goals

1. **Rebuild GPT-2-Small**: Implement a transformer architecture from scratch with 124M parameters
2. **Large-Scale Pretraining**: Train on the 10B token FineWeb-Edu dataset to establish general linguistic competence
3. **Benchmark Performance**: Evaluate the model on HellaSwag to verify it learned meaningful language patterns
4. **Domain-Specific Fine-tuning**: Adapt the pretrained model for fantasy location descriptions, proving that small models can excel in narrow domains when properly specialized

## Performance Summary

### Pretraining Results
- **Initial Training Run**: 19,073 steps (~1 epoch over 10B tokens), reaching 30.52% HellaSwag accuracy
- **Extended Training Run**: 57,219 steps (~3 epochs), reaching 32.5% HellaSwag accuracy
- **Final Validation Loss**: Continued to decrease throughout training, indicating the model remained undertrained
- **Batch Size**: 524,288 tokens per step (effective batch via gradient accumulation)
- **Hardware**: 1x H100 80GB GPU via Lambda Cloud (~450,000 tokens/sec, 1000x faster than CPU)
- **Training Cost**: ~$21.40 for 1-epoch run, ~$64.50 for 3-epoch run

### HellaSwag Benchmark
- **1-Epoch Accuracy**: 30.52% (3,065/10,042 correct)
- **3-Epoch Accuracy**: 32.53% (3,267/10,042 correct)
- **Baseline**: OpenAI GPT-2 Small achieves 28.9%
- **Result**: Exceeded baseline by ~3.6 percentage points, approaching GPT-3 Small's 33% despite using only 10B tokens (vs. 100B+ for GPT-2/3)

### Fine-tuning Results
- **Dataset**: 1,000 synthetic fantasy location descriptions generated with GPT-4/GPT-5 (~359k tokens)
- **Best Hyperparameter**: Learning rate 2e-4 (from search over 6 values: 1e-4 to 2e-3)
- **Training Progress**: Loss decreased from 3.84 → 0.32 over 7 epochs
- **Validation**: Best checkpoint at epoch 2 before overfitting began
- **HellaSwag After Fine-tuning**: 28.9% (expected drop as model specializes away from general language)
- **Output Quality**: Generates coherent, atmospheric fantasy descriptions with rich sensory details, strong stylistic consistency, and evocative imagery. While not perfect, outputs are suitable for dynamic worldbuilding in games or tabletop RPG assistants.

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

### Step 3: Evaluate on HellaSwag Benchmark

After pretraining, evaluate the model's commonsense reasoning ability:

```bash
python run_hellaswag.py
```

This script:
- Downloads the HellaSwag validation dataset (10,042 examples)
- Loads the pretrained model checkpoint
- Evaluates accuracy on multiple-choice sentence completion
- Reports overall accuracy percentage

### Step 4: Generate Fine-tuning Dataset

Generate synthetic training data for domain-specific fine-tuning:

```bash
python generate_finetune_data.py
```

This script:
- Uses OpenAI's GPT-4.1 API to generate 1,000 fantasy location descriptions
- Each description is 150-400 words with atmospheric, sensory details
- Saves data to `datasets/finetune_data.txt` with `<LOC>...</LOC>` delimiters
- Total dataset: ~359k tokens (1.6 MB)

**Note**: Requires OpenAI API key set in the script or environment.

### Step 5: Fine-tune the Model

Fine-tune the pretrained model on the fantasy location dataset:

```bash
python fine_qlora.py
```

The fine-tuning script:
- **Phase 1**: Hyperparameter search over 6 learning rates (1e-4 to 7e-3)
- **Phase 2**: Full training with best learning rate (2e-4) for 7 epochs
- Uses packed sequence dataset for efficiency (512 token sequences)
- Batch size of 8 with gradient accumulation of 8
- Saves checkpoints after each epoch to `finetune_runs/final_lr_0.0002/`
- Train/val split: 630/70 sequences (90%/10%)

### Step 6: Generate Samples from Fine-tuned Model

Generate fantasy location descriptions using the fine-tuned model:

```bash
python gen_after_finetuny.py
```

This script:
- Loads the fine-tuned checkpoint
- Uses top-K sampling (K=50) with temperature 0.9
- Generates 300 token samples
- Saves results to `final_gens_after_finetune.txt`

## Project Structure

### Core Model Components
- `gpt_model.py` - GPT-2 architecture implementation (124M parameters, 12 layers, 768 dimensions)
- `transformer_modules.py` - Transformer building blocks (CausalSelfAttention, MLP, Block)
- `data_loader.py` - Efficient sharded data loading with automatic shard rotation

### Training Scripts
- `fineweb_edu.py` - Dataset download, tokenization, and sharding for FineWeb-Edu 10B
- `train_gpt2.py` - Pretraining loop with cosine LR decay, gradient accumulation, mixed precision
- `generate_finetune_data.py` - Synthetic data generation using GPT-4.1 for fantasy locations
- `fine_qlora.py` - Fine-tuning with hyperparameter search and full model training

### Evaluation and Generation
- `hellaswag.py` - HellaSwag benchmark utilities
- `run_hellaswag.py` - Evaluation script for pretrained model
- `gen_after_finetuny.py` - Sample generation from fine-tuned model

### Datasets
- `datasets/finetune_data.txt` - 1,000 fantasy location descriptions (1.6 MB)
- `datasets/tiny-shakespeare.txt` - Small dataset for initial testing
- `datasets/edu_fineweb10B/` - Sharded 10B token dataset (generated by fineweb_edu.py)

### Model Artifacts
- `training_run/log/` - Pretraining checkpoints and logs (4 checkpoints at steps 5k, 10k, 15k, 19k)
- `finetune_runs/final_lr_0.0002/` - Fine-tuning checkpoints (7 epochs)
- `hellaswag/hellaswag_val.jsonl` - HellaSwag benchmark dataset

### Results
- `final_gens_after_finetune.txt` - Sample generations from fine-tuned model
- `finetune_hyperparameter_search.txt` - Learning rate search results
- `training_run/log/log.txt` - Pretraining loss curve

## Technical Highlights

### Modern Optimizations
- Flash attention via `F.scaled_dot_product_attention`
- Mixed precision training with bfloat16
- Gradient accumulation for large effective batch sizes
- `torch.compile` for improved training speed
- Fused AdamW optimizer

### Architecture Fidelity
- Exact GPT-2 Small architecture (12 layers, 12 heads, 768 dimensions)
- Proper weight initialization matching OpenAI's implementation
- Weight sharing between token embeddings and output layer
- Residual connection scaling for deep networks

### Training Best Practices
- Cosine learning rate schedule with 200-step linear warmup
- Gradient clipping at 1.0
- Regular validation evaluation (every 100 steps)
- Checkpoint saving for reproducibility
- Separate train/validation splits

## Results and Insights

### Key Findings

**Small models can excel at specialized tasks**: The 124M-parameter model, though 100x smaller than modern LLMs, successfully generates rich fantasy descriptions after domain-specific fine-tuning. This demonstrates that specialized small models are viable for resource-constrained applications.

**High-quality data matters more than quantity**: Using the FineWeb-Edu dataset (curated for educational content) enabled the model to exceed GPT-2's baseline performance despite training on 10x fewer tokens than the original GPT-2.

**Fine-tuning creates strong stylistic specialization**: With only ~359k tokens of fantasy descriptions, the model learned to consistently produce atmospheric, sensory-rich text matching the target style. The dramatic qualitative shift from generic to evocative prose validates the fine-tuning approach.

**Efficient training is achievable**: Modern optimizations (flash attention, mixed precision, torch.compile) made training practical on a single GPU, with total costs under $70 for 3 epochs of pretraining.

### Observations
- Validation loss continued decreasing throughout pretraining, indicating the model remained undertrained and would benefit from more compute
- Fine-tuning showed overfitting after epoch 2, as expected for a small specialized corpus
- Generated samples exhibit strong stylistic consistency but occasional mid-generation coherence loss
- HellaSwag accuracy improved from 30.5% (1 epoch) to 32.5% (3 epochs), confirming continued learning

### Limitations and Future Work
- **Data ordering effects**: Loss curve shows periodic spikes due to unshuffled training shards
- **Context length**: Increasing sequence length during fine-tuning could enable longer descriptions
- **Model size scaling**: Exploring whether larger models (e.g., 350M or 774M parameters) improve quality while remaining deployable
- **Controlled generation**: Adding control tokens for theme, tone, or location type would increase practical utility

### Academic Achievement
This project demonstrates the complete deep learning lifecycle from scratch: architecture implementation, large-scale pretraining on 10B tokens, rigorous evaluation on standard benchmarks, and targeted domain adaptation. The results prove that smaller, specialized models remain a viable design space for applications where speed, cost, and deployment constraints matter more than general-purpose reasoning.

## License

Academic project for CU Boulder MSCS Deep Learning course.
