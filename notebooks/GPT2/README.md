# GPT-2 Implementation from Scratch

A complete educational implementation of GPT-2 built from scratch using PyTorch. Each notebook focuses on a specific component of the transformer architecture with detailed explanations and verbose debugging output.

## Notebooks

| Notebook | Content | Key Components |
|----------|---------|----------------|
| **01. DataPreparation** | Text preprocessing and tokenization | GPTDataset, DataLoader, Token embeddings |
| **02. MultiHeadAttention** | Attention mechanism | Causal self-attention, Multi-head attention, Scaled dot-product |
| **03. Normalization** | Layer normalization | LayerNorm with learnable scale/shift |
| **04. FeedForward** | Feed-forward network | GELU activation, 4x expansion MLP |
| **05. TransformerBlock** | Complete transformer block | Pre-norm architecture, Residual connections |
| **06. GPTModel** | Full GPT model | Token generation, Autoregressive decoding |
| **07. Loss, Cross Entropy** | Loss calculation and evaluation | Cross-entropy loss, Perplexity, Manual vs PyTorch comparison |
| **08. Training** | Model training loop | Training/validation split, Optimizer, Weight updates |

## Architecture

**Standard GPT-2 transformer architecture:**
1. Token + positional embeddings
2. N transformer blocks (self-attention → feed-forward with residual connections)
3. Final layer normalization → output projection to vocabulary

**Default configuration (GPT-2 124M):**
```python
{
    "vocab_size": 50257,      # GPT-2 vocabulary
    "context_length": 256,    # Max sequence length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Attention heads
    "n_layers": 12,           # Transformer blocks
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}
```

## Key Features

- **Educational focus**: Clear explanations with verbose debugging output
- **Modular design**: Each component in separate notebook
- **Manual implementations**: Loss calculation, embeddings, attention from scratch
- **Validation**: Compare custom implementations against PyTorch built-ins
- **Visualization**: Training curves, activation functions, loss vs perplexity

## Usage

Notebooks are designed to be run sequentially. Each imports and reuses components from previous notebooks using `%run`.

```python
%run "01. DataPreparation.ipynb"  # Import dataset utilities
%run "06. GPTModel.ipynb"         # Import model
```

## Reference

This implementation is based on concepts from:

**Raschka, Sebastian.** *Build a Large Language Model (From Scratch)*. Manning Publications, 2024.
ISBN: 978-1633437166
GitHub: https://github.com/rasbt/LLMs-from-scratch

## Dependencies

- PyTorch
- tiktoken (GPT-2 tokenizer)
- matplotlib (visualizations)

See `requirements.txt` for complete list.