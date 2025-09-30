# GPT2 Implementation Notebooks

This folder contains a complete implementation of a GPT-2 model built from scratch using PyTorch. Each notebook focuses on a specific component of the transformer architecture.

## Notebooks Overview

### 01. DataPreparation.ipynb
**Purpose**: Data preprocessing and tokenization
- **GPTDataset**: Custom dataset class that tokenizes text and creates training pairs using sliding window approach
- **DataLoader**: Wrapper function for batch creation with configurable stride and context length
- **Embedder**: Demonstration token and positional embedding implementation
- **Key Features**: Uses tiktoken for GPT-2 compatible tokenization, creates input-target pairs shifted by 1 position

### 02. MultiHeadAttention.ipynb
**Purpose**: Attention mechanism implementation
- **CausalSelfAttention**: Single attention head with causal masking
- **MultiHeadAttention**: Complete multi-head attention with verbose debugging output
- **Key Features**: Scaled dot-product attention, causal masking, dropout, head concatenation
- **Detailed Logging**: Step-by-step attention computation with intermediate results

### 03. Normalization.ipynb
**Purpose**: Layer normalization implementation
- **LayerNorm**: Custom layer normalization with learnable scale and shift parameters
- **Key Features**: Normalizes to mean=0, variance=1, with epsilon for numerical stability
- **Debugging**: Verbose output showing normalization statistics

### 04. FeedForward.ipynb
**Purpose**: Feed-forward network with GELU activation
- **GELU**: Custom implementation of Gaussian Error Linear Unit activation
- **FeedForward**: Two-layer MLP with 4x hidden dimension expansion
- **Visualization**: Comparison plots between GELU and ReLU activation functions

### 05. TransformerBlock.ipynb
**Purpose**: Complete transformer block assembly
- **TransformerBlock**: Combines attention, normalization, and feed-forward layers
- **Architecture**: Pre-norm design with residual connections
- **Key Features**: Dropout, skip connections, detailed forward pass logging

### 06. GPTModel.ipynb
**Purpose**: Complete GPT model implementation
- **GPTModel**: Full transformer model with embedding layers and output head
- **Token Generation**: Implements autoregressive text generation
- **Key Features**:
  - Token and positional embeddings
  - Multiple transformer blocks in sequence
  - Final normalization and output projection
  - Configurable model parameters (124M parameter config included)

## Model Architecture

The implementation follows the standard GPT-2 architecture:
1. **Input Processing**: Token + positional embeddings with dropout
2. **Transformer Blocks**: N layers of self-attention + feed-forward with residual connections
3. **Output**: Final layer norm + linear projection to vocabulary

## Key Features

- **Verbose Debugging**: All components include detailed logging for educational purposes
- **Causal Masking**: Ensures autoregressive behavior in attention
- **Configurable**: Model size and hyperparameters easily adjustable
- **Educational Focus**: Clear separation of concerns with extensive documentation

## Configuration

Default GPT-2 124M parameter configuration:
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

## Dependencies

See `requirements.txt` for required packages.