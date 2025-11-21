# GPT-2 From Scratch: Deep Learning Final Project

Final project for the CU Boulder MSCS Deep Learning course. This project demonstrates the complete lifecycle of building, training, and fine-tuning a large language model from scratch. The project implements GPT-2-Small (124M parameters), trains it on 10 billion tokens of web data, benchmarks its performance, and fine-tunes it for fantasy location descriptions to show that smaller, specialized models can excel in narrow domains.

## Project Goals

1. **Rebuild GPT-2-Small**: Implement a transformer architecture similar to GPT-2-Small (124M parameters) from the ground up
2. **Large-Scale Pretraining**: Train the model on the 10B token FineWeb-Edu dataset
3. **Benchmark Performance**: Evaluate the pretrained model on HellaSwag commonsense reasoning benchmark
4. **Domain-Specific Fine-tuning**: Fine-tune the pretrained model for fantasy location descriptions to demonstrate domain specialization

## Performance Summary

### Pretraining Results
- **Training Duration**: 19,073 steps (~1 epoch over 10B tokens)
- **Final Validation Loss**: 61.75 (down from initial 219.94)
- **Batch Size**: 524,288 tokens per step (64 micro-batches × 1024 sequence length × 8 gradient accumulation)
- **Training Time**: Multiple hours on CUDA GPU

### HellaSwag Benchmark
- **Accuracy**: 30.52% (3,065/10,042 correct)
- **Baseline**: OpenAI GPT-2 Small achieves 28.9%
- **Result**: Successfully exceeded baseline performance despite training from scratch

### Fine-tuning Results
- **Dataset**: 1,000 synthetic fantasy location descriptions (~359k tokens)
- **Best Hyperparameter**: Learning rate 2e-4 (from search over 6 values)
- **Training Progress**: Loss decreased from 3.84 → 0.32 over 7 epochs
- **Validation**: Best epoch was 2 (val_loss=3.49) before overfitting began
- **Output Quality**: Generates coherent, atmospheric fantasy descriptions with rich sensory details

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

### Strengths
- Successfully trains GPT-2 from scratch and exceeds baseline HellaSwag performance
- Clean, well-structured code with proper abstractions
- Comprehensive pipeline from data preparation to evaluation
- Effective domain adaptation through fine-tuning

### Observations
- Fine-tuning shows overfitting after epoch 2 (validation loss increases while training loss decreases)
- Generated samples are coherent with atmospheric language but show some repetitive patterns
- Model size (124M parameters) is appropriate for academic demonstration
- Hyperparameter search successfully identified optimal learning rate

### Academic Achievement
This project demonstrates the complete machine learning pipeline: architecture design, large-scale training, benchmarking, and domain-specific fine-tuning. The model's HellaSwag performance (30.52%) exceeding the baseline (28.9%) validates the training approach, while the fine-tuned model generates high-quality fantasy descriptions with rich sensory details.

## License

Academic project for CU Boulder MSCS Deep Learning course.
