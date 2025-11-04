# Time Series Forecasting with Transformer

A PyTorch-based time series forecasting model that uses transformer architecture with tokenization to predict future values in temporal sequences. The model treats time series forecasting as a sequence modeling problem, similar to language modeling.

## Overview

This project implements a transformer-based forecasting model that:
- **Tokenizes** continuous time series values into discrete bins
- **Uses multi-query attention** for efficient sequence processing
- **Supports autoregressive generation** for multi-step forecasting
- **Leverages pre-trained models** from HuggingFace's GiftEvalPretrain dataset

The approach is inspired by recent work on treating time series as sequences of tokens, allowing the model to learn patterns and dependencies similar to how language models process text.

## Architecture

### Model Components

The forecasting model consists of several key components:

#### 1. **Tokenizer** (`src/tokenizer.py`)
Converts continuous time series values into discrete tokens:
- Scales values using mean absolute value normalization
- Bucketizes scaled values into predefined bins (vocabulary)
- Handles padding for sequences shorter than context length
- Supports inverse transformation for decoding predictions

#### 2. **ForecastingModel** (`src/model.py`)
The main transformer-based model with:
- **Token Embeddings**: Maps discrete tokens to continuous representations
- **Positional Embeddings**: Adds temporal position information using sinusoidal encoding
- **Transformer Blocks**: Multiple layers of attention and feed-forward networks
- **Output Projection**: Maps hidden states back to vocabulary logits
- **Temperature Parameter**: Learnable parameter for output scaling

#### 3. **Multi-Query Attention** (`src/attention.py`)
Efficient attention mechanism where:
- Multiple query heads share the same key and value projections
- Reduces memory and computation while maintaining expressiveness
- Implements causal masking for autoregressive generation
- Supports KV-caching for efficient inference

#### 4. **Feed-Forward Network** (`src/ffn.py`)
Standard two-layer MLP with:
- ReLU activation
- Configurable hidden dimension
- Dropout for regularization

#### 5. **Layer Normalization** (`src/normalization.py`)
Custom implementation of layer normalization with learnable scale and shift parameters.

#### 6. **Positional Embedding** (`src/positional.py`)
Sinusoidal positional encodings that inject position information into the model.

### Model Configuration

Key hyperparameters (defined in `src/config.py`):
- `block_size`: Context window size (default: 256)
- `vocab_size`: Number of discrete bins for tokenization (default: 64)
- `n_layer`: Number of transformer blocks (default: 6)
- `n_head`: Number of attention heads (default: 6)
- `n_embd`: Embedding dimension (default: 384)
- `dropout`: Dropout rate (default: 0.2)
- `ffn_hidden_size`: Feed-forward hidden dimension (default: 256)

## Installation

### Prerequisites
- Python >= 3.12
- PyTorch >= 2.9.0

### Setup

```bash
# Clone the repository
git clone https://github.com/Bornlex/timeseries-forecasting.git
cd timeseries-forecasting

# Install dependencies
pip install -r requirements.txt
# or if using uv:
uv sync
```

### Dependencies
- `torch>=2.9.0` - Deep learning framework
- `datasets>=4.3.0` - HuggingFace datasets for loading time series data
- `seaborn>=0.13.2` - Visualization
- `wandb>=0.22.3` - Experiment tracking
- `pytest>=8.4.2` - Testing framework

## Usage

### 1. Data Exploration

Explore and visualize datasets from HuggingFace's GiftEvalPretrain collection:

```bash
# Display available dataset names
python explore.py --display_names

# Visualize a specific dataset
python explore.py --dataset_name BEIJING_SUBWAY_30MIN --num_series 5

# Extract and save subsets for training
python explore.py --output_file dataset.jsonl --extract_per_split 1000
```

**Key arguments:**
- `--dataset_name`: Name of the dataset to explore (default: 'BEIJING_SUBWAY_30MIN')
- `--num_series`: Number of series to plot (default: 5)
- `--display_names`: Show all available dataset names
- `--output_file`: Save extracted data to JSONL file
- `--extract_per_split`: Number of series per split to extract (default: 1000)

### 2. Training

Train a forecasting model on your time series data:

```bash
# Basic training
python training.py --dataset dataset.jsonl --max_iters 5000 --wandb_log

# Training with custom hyperparameters
python training.py \
    --dataset dataset.jsonl \
    --n_layers 4 \
    --n_head 8 \
    --n_embd 512 \
    --n_ffn 2048 \
    --vocab_size 1024 \
    --block_size 256 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --max_iters 10000 \
    --save \
    --save_path model.pth

# Resume training from checkpoint
python training.py --resume --resume_path checkpoint.pth
```

**Key training arguments:**

*Model Architecture:*
- `--n_layers`: Number of transformer layers (default: 2)
- `--n_head`: Number of attention heads (default: 2)
- `--n_embd`: Embedding dimension (default: 384)
- `--n_ffn`: Feed-forward hidden size (default: 1024)
- `--vocab_size`: Number of discrete bins (default: 1024)
- `--dropout`: Dropout rate (default: 0.2)

*Data & Training:*
- `--dataset`: Path to dataset JSONL file (default: 'dataset.jsonl')
- `--batch_size`: Batch size (default: 64)
- `--block_size`: Context window size (default: 256)
- `--max_iters`: Total training iterations (default: 5000)
- `--num_series`: Number of series to train on (default: 2000)

*Optimization:*
- `--learning_rate`: Initial learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for AdamW (default: 1e-1)
- `--beta1`: AdamW beta1 (default: 0.9)
- `--beta2`: AdamW beta2 (default: 0.99)
- `--lr_decay_iters`: Iterations for cosine annealing (default: 5000)
- `--min_lr`: Minimum learning rate (default: 1e-4)

*Tokenization:*
- `--low_limit`: Lower bound for value scaling (default: -15.0)
- `--high_limit`: Upper bound for value scaling (default: 15.0)

*Logging:*
- `--wandb_log`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name (default: 'time-series-forecasting')
- `--eval_interval`: Iterations between evaluations (default: 250)
- `--log_interval`: Iterations between logging (default: 10)

*Checkpointing:*
- `--save`: Save final model
- `--save_path`: Path to save trained model (default: 'model.pth')
- `--resume`: Resume from checkpoint
- `--resume_path`: Path to checkpoint (default: 'checkpoint.pth')

**Training Process:**

The training loop:
1. Loads time series data from JSONL file
2. Creates sliding windows over each series
3. Tokenizes values into discrete bins
4. Trains model with cross-entropy loss
5. Evaluates by generating forecasts on validation set
6. Logs metrics to W&B (loss, MSE, MAE, learning rate)
7. Saves checkpoints periodically
8. Generates visualization plots during training

### 3. Inference

Generate forecasts from a trained model:

```bash
# Basic inference
python inference.py --model_path checkpoint.pth --dataset dataset.jsonl

# Inference with custom generation parameters
python inference.py \
    --model_path model.pth \
    --dataset dataset.jsonl \
    --n_layers 4 \
    --n_head 8 \
    --n_embd 512 \
    --vocab_size 1024 \
    --block_size 256
```

**Key inference arguments:**
- `--model_path`: Path to trained model (default: 'checkpoint.pth')
- `--dataset`: Path to dataset for sampling input series (default: 'dataset.jsonl')
- `--seed`: Random seed for reproducibility (default: 42)

*Model Configuration (must match training):*
- `--n_layers`: Number of transformer layers (default: 2)
- `--n_head`: Number of attention heads (default: 2)
- `--n_embd`: Embedding dimension (default: 384)
- `--vocab_size`: Vocabulary size (default: 1024)
- `--block_size`: Context window size (default: 256)

**Inference Process:**

The inference script:
1. Loads a trained model from checkpoint
2. Samples a random series from the dataset
3. Generates 256 future tokens autoregressively
4. Uses temperature, top-k, and top-p sampling
5. Visualizes observed values vs. forecast

### 4. Generation Module

The generation module (`src/generation.py`) provides flexible autoregressive forecasting:

```python
from src.generation import generate_from_series

# Generate forecast
forecast = generate_from_series(
    series=your_series,          # List of observed values
    model=trained_model,         # Trained ForecastingModel
    tokenizer=tokenizer,         # Tokenizer instance
    max_tokens=256,              # Number of steps to forecast
    temperature=0.8,             # Sampling temperature (higher = more random)
    top_k=10,                    # Keep top-k tokens for sampling
    top_p=0.9,                   # Nucleus sampling threshold
    log_file='generation.json'   # Save generation statistics
)
```

**Generation Parameters:**
- `temperature`: Controls randomness (0.1-2.0). Lower = more conservative
- `top_k`: Number of top tokens to consider (0 = disabled)
- `top_p`: Cumulative probability threshold for nucleus sampling
- `log_file`: Optional path to save detailed generation logs (token probs, entropy, etc.)

## Project Structure

```
timeseries-forecasting/
├── src/
│   ├── attention.py         # Multi-query attention implementation
│   ├── config.py            # Model configuration dataclass
│   ├── ffn.py              # Feed-forward network
│   ├── generation.py        # Autoregressive generation utilities
│   ├── model.py            # Main forecasting model
│   ├── normalization.py    # Layer normalization
│   ├── positional.py       # Positional embeddings
│   ├── tokenizer.py        # Time series tokenization and data loading
│   └── visualisation/
│       └── chart.py        # Plotting utilities
├── tests/
│   └── test_tokenization.py  # Unit tests for tokenization
├── training.py             # Training script
├── inference.py            # Inference script
├── explore.py              # Data exploration script
├── pyproject.toml          # Project dependencies
└── README.md              # This file
```

## Key Concepts

### Tokenization

Time series values are converted to discrete tokens through:
1. **Normalization**: Divide by mean absolute value of the series
2. **Bucketization**: Map normalized values to bins using predefined boundaries
3. **Special Tokens**: Reserve token IDs for padding and special markers

This allows the model to learn relationships between discrete value ranges rather than raw continuous values.

### Autoregressive Generation

Forecasting is performed autoregressively:
1. Start with observed context (last `block_size` tokens)
2. Model predicts probability distribution over next token
3. Sample from distribution using temperature/top-k/top-p
4. Append sampled token to context
5. Repeat for desired forecast horizon

### Training Strategy

- **Loss**: Cross-entropy on token predictions
- **Optimization**: AdamW with cosine annealing
- **Regularization**: Dropout, weight decay, gradient clipping
- **Evaluation**: MSE/MAE on token IDs, visual inspection of forecasts
- **Monitoring**: W&B integration for tracking metrics and artifacts

## Data Format

Input data should be in JSONL format with the following structure:

```json
{"target": [[value1, value2, value3, ...]], "other_metadata": "..."}
{"target": [[value1, value2, value3, ...]], "other_metadata": "..."}
```

Each line contains a JSON object with at least a `target` field containing the time series as a nested list.

## Testing

Run unit tests:

```bash
pytest tests/
```

The test suite includes:
- Tokenization correctness
- Window pointer generation
- Data loading pipeline
- Handling of NaN values

## Model Checkpoints

Models are saved as PyTorch state dictionaries:
- **Training checkpoints**: Saved periodically to `checkpoint.pth`
- **Final models**: Saved with `--save` flag to specified path
- **Loading**: Use `model.load_state_dict(torch.load(path))`

## Device Support

The code automatically detects and uses:
- **Apple Silicon (MPS)**: For M1/M2 Macs
- **CPU**: Fallback for systems without GPU

To add CUDA support, modify device selection in training and inference scripts.

## Visualization

The project includes visualization tools:
- **Training plots**: Generated during training to `img/` directory
- **Forecast plots**: Compare observed vs predicted values
- **Data exploration**: Visualize multiple series in subplots

## Tips for Best Results

1. **Data Quality**: Clean data with minimal NaN values performs best
2. **Context Length**: Longer context (256-512) captures more patterns
3. **Vocabulary Size**: Balance between granularity (large vocab) and generalization (small vocab)
4. **Value Scaling**: Adjust `low_limit` and `high_limit` based on data range
5. **Hyperparameter Tuning**: Use W&B sweeps to find optimal settings
6. **Regularization**: Increase dropout for overfitting, decrease for underfitting

## Contributing

Contributions are welcome! Areas for improvement:
- CUDA support
- Additional attention mechanisms
- Multi-variate forecasting
- Probabilistic forecasting (quantile prediction)
- Pre-trained model zoo
- More comprehensive evaluation metrics

## License

[Add your license information here]

## Acknowledgments

This project uses:
- PyTorch for deep learning
- HuggingFace Datasets for data loading
- Weights & Biases for experiment tracking
- The GiftEvalPretrain dataset for training data
