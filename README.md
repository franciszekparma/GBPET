# GBPET

**GPT with Byte Pair Encoding Tokenizer**

A decoder-only transformer language model with a custom Byte Pair Encoding (BPE) tokenizer, built entirely from scratch in PyTorch. Trained on 15 million characters of Charles Dickens novels.

No external ML libraries. No HuggingFace. No pretrained models. No SentencePiece or tiktoken. Everything implemented from first principles.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Tokenization System](#tokenization-system)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Hardware](#hardware)
- [Results](#results)
- [Sample Output](#sample-output)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

GBPET implements a GPT-style autoregressive language model that generates coherent, period-appropriate prose in the style of Charles Dickens. The project demonstrates a complete understanding of modern language model architecture by implementing every component from scratch:

- **Custom BPE tokenizer** implementing the original algorithm from Sennrich et al. (2016)
- **Decoder-only transformer** with multi-head self-attention, feed-forward networks, and learned positional embeddings
- **Dual tokenization modes**: Switch between BPE (subword) and character-level tokenization via a single flag
- **Bigram baseline model** for performance comparison
- **Checkpoint system** that preserves model weights, optimizer state, scheduler state, and the complete tokenizer vocabulary

The model is trained on a corpus of Dickens novels (public domain texts), totaling approximately 15 million characters.

---

## Features

| Feature | Description |
|---------|-------------|
| **From-Scratch Implementation** | No external ML libraries — pure PyTorch implementation |
| **Custom BPE Tokenizer** | Full Byte Pair Encoding algorithm with vocabulary learning |
| **Dual Tokenization** | Toggle between BPE (subword) and character-level modes |
| **Pre-Norm Architecture** | GPT-2 style LayerNorm placement for stable training |
| **Learned Positional Embeddings** | Trainable position representations |
| **Comprehensive Checkpointing** | Saves model, optimizer, scheduler, and full tokenizer state |
| **Clean Text Generation** | Post-processing for proper spacing and punctuation |
| **Baseline Comparison** | Bigram model included for benchmarking |

---

## Model Architecture

### Specifications

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | ~20M |
| **Embedding Dimension** | 512 |
| **Attention Heads** | 8 |
| **Transformer Blocks** | 8 |
| **Head Dimension** | 64 |
| **Context Length** | 256 tokens |
| **Vocabulary Size** | 2048 (BPE) / ~88 (char-level) |
| **Feed-Forward Dimension** | 2048 (4 × emb_dim) |
| **Dropout** | 0.5 |

### Design Choices

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| **Normalization** | Pre-LayerNorm | Improves training stability (GPT-2 style) |
| **Activation** | GELU | Smoother than ReLU, standard in transformers |
| **Positional Encoding** | Learned embeddings | More flexible than sinusoidal |
| **Attention Mask** | Causal (lower triangular) | Prevents attending to future tokens |

### Architecture Diagram

```
Input Token IDs
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Token Embedding                           │
│                nn.Embedding(vocab_size, 512)                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  + Positional Embedding                      │
│              nn.Embedding(256, 512) [learned]                │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
╔══════════════════════════════════════════════════════════════╗
║                  TRANSFORMER BLOCK (×8)                      ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │                 LayerNorm (Pre-Norm)                   │  ║
║  └────────────────────────┬───────────────────────────────┘  ║
║                           │                                  ║
║                           ▼                                  ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │           Multi-Head Self-Attention                    │  ║
║  │           (8 heads, causal masking)                    │  ║
║  │                                                        │  ║
║  │   Q, K, V = Linear(512, 512) each                      │  ║
║  │   Attention = softmax(QK^T / √64) × V                  │  ║
║  │   Output = Linear(512, 512)                            │  ║
║  └────────────────────────┬───────────────────────────────┘  ║
║                           │                                  ║
║                      + Residual                              ║
║                           │                                  ║
║                           ▼                                  ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │                 LayerNorm (Pre-Norm)                   │  ║
║  └────────────────────────┬───────────────────────────────┘  ║
║                           │                                  ║
║                           ▼                                  ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │              Feed-Forward Network                      │  ║
║  │                                                        │  ║
║  │   Linear(512 → 2048) → GELU → Linear(2048 → 512)       │  ║
║  └────────────────────────┬───────────────────────────────┘  ║
║                           │                                  ║
║                      + Residual                              ║
║                           │                                  ║
╚═══════════════════════════╪══════════════════════════════════╝
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    Final LayerNorm                           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                Linear Head (512 → vocab_size)                │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                     Output Logits
```

---

## Tokenization System

The project supports two tokenization modes, configurable via the `USE_BYTE_PAIR` flag in `utils.py`.

### Byte Pair Encoding (BPE)

The BPE tokenizer is implemented entirely from scratch, following the original algorithm from [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909).

#### Algorithm

1. **Corpus Preprocessing**: Text is split into words at spaces. Punctuation and special characters (`\n`, `\t`, `\r`) are treated as separate tokens. Each word ends with a special `</w>` end-of-word marker.

2. **Vocabulary Initialization**: The initial vocabulary consists of all unique characters in the corpus plus `</unk>` for unknown tokens (~88 base tokens).

3. **Merge Learning**: The algorithm iteratively:
   - Finds the most frequent adjacent token pair across the corpus
   - Merges them into a new token
   - Adds the new token to the vocabulary
   - Repeats until `TARGET_VOCAB_SIZE` (2048) is reached

4. **Encoding**: Text is first tokenized into characters with `</w>` markers, then all learned merges are applied in order. Each token is mapped to an integer ID via the `st_2_i` dictionary.

5. **Decoding**: Integer IDs are mapped back to tokens via the `i_2_st` dictionary, concatenated, and `</w>` is replaced with spaces.

#### Post-Processing

The `clean_decode()` method fixes common spacing artifacts:

| Before | After |
|--------|-------|
| `don ' t` | `don't` |
| `word .` | `word.` |
| `" hello "` | `"hello"` |
| `( example )` | `(example)` |

### Character-Level Tokenization

Simple baseline where each unique character (~88) is a token. Toggle with `USE_BYTE_PAIR = False` in `utils.py`.

**Trade-off**: Simpler implementation but requires longer sequences to represent the same text, making training less efficient.

---

## Project Structure

```
GBPET/
│
├── GBPET/
│   ├── utils.py              # Configuration and hyperparameters
│   │   ├── Model hyperparameters (EMB_DIM, N_HEADS, N_BLOCKS...)
│   │   ├── Training settings (LR, BATCH_SIZE, EPOCHS...)
│   │   ├── Checkpoint paths and flags
│   │   └── USE_BYTE_PAIR toggle
│   │
│   ├── data_preparation.py   # Tokenization and data loading
│   │   ├── BytePairEncoding class
│   │   │   ├── train()         — Learn BPE merges from corpus
│   │   │   ├── encode()        — Text → token IDs
│   │   │   ├── decode()        — Token IDs → text
│   │   │   └── clean_decode()  — Fix spacing artifacts
│   │   └── Sample_Batches class — Batch sampling for training
│   │
│   ├── transformer.py        # Model architecture and training
│   │   ├── Head                — Single attention head
│   │   ├── MultiHeadAttention  — Parallel attention heads
│   │   ├── FeedForward         — MLP block with GELU
│   │   ├── Block               — Complete transformer block
│   │   ├── Language_Model      — Full GPT model
│   │   └── Training loop with checkpointing
│   │
│   └── bigram.py             # Baseline model for comparison
│       └── Bigram_LM           — Simple next-token prediction
│
├── data/
│   └── dickens_corpus.txt    # ~15M characters of Dickens novels
│
├── checkpoints/              # Saved model checkpoints
│
├── samples/                  # Generated text samples
│
└── LICENSE                   # MIT License
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

### Setup

```bash
# Clone the repository
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET

# Install dependencies
pip install torch numpy tqdm
```

---

## Usage

### Training from Scratch

```bash
cd GBPET
python transformer.py
```

This will:
1. Load the Dickens corpus
2. Train the BPE tokenizer (if not cached)
3. Encode the corpus
4. Train the transformer model
5. Save checkpoints on validation improvement
6. Generate sample text after training

### Generate Text Only

To generate text from a trained checkpoint:

```python
# In utils.py, set:
ONLY_GENERATE = True
TRS_LOAD_CHECKPOINT = True
```

Then run:
```bash
python transformer.py
```

### Using Pre-encoded Corpus

To skip BPE training and use a pre-computed vocabulary:

```python
# In utils.py, set:
LOAD_BPE_ONLY = True
TRS_LOAD_PATH = "path/to/checkpoint.pt"
```

### Baseline Comparison

Run the bigram baseline model:

```bash
python bigram.py
```

---

## Training Configuration

### Optimizer

| Parameter | Value |
|-----------|-------|
| **Algorithm** | AdamW |
| **Learning Rate** | 6e-4 |
| **Min Learning Rate** | 1e-6 |
| **Gradient Clipping** | max_norm = 1.0 |

### Learning Rate Schedule

| Phase | Strategy |
|-------|----------|
| **Warmup** | Linear warmup for first 5% of steps (~1,638 steps) |
| **Decay** | Cosine annealing to min_lr |

```
Learning Rate
     ↑
6e-4 │        ╭─────────────╮
     │       ╱               ╲
     │      ╱                 ╲
     │     ╱                   ╲
1e-6 │────╱                     ╲────────
     └────┬──────────┬──────────┬──────→ Steps
          0       1,638      32,768
           Warmup    Cosine Decay
```

### Data Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 256 |
| **Context Length** | 256 tokens |
| **Train/Val Split** | 80% / 20% |
| **Epochs** | 128 |
| **Batches per Epoch** | 256 |
| **Total Steps** | 32,768 |

### Checkpointing

Checkpoints are saved when validation loss improves and include:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current epoch and best validation loss
- Full BPE vocabulary (`merges`, `st_2_i`, `i_2_st`)
- Encoded corpus and train/val splits
- Architecture configuration for easy loading

---

## Hardware

| Tokenization Mode | GPU | Cloud Provider |
|-------------------|-----|----------------|
| **BPE (subword)** | NVIDIA RTX 5090 | Rented (RunPod) |
| **Character-level** | NVIDIA RTX 4090 | Rented (RunPod) |

### Training Time

| Component | Duration |
|-----------|----------|
| BPE vocabulary training | ~3-4 hours (one-time, cached) |
| Model training per epoch | ~100-120 seconds |
| Full training run | ~3-4 hours |

**Note**: BPE vocabulary training runs on CPU. Once computed, it's saved to checkpoint and loaded instantly on subsequent runs.

---

## Results

### Loss Metrics

| Metric | BPE Model | Character Model |
|--------|-----------|-----------------|
| **Final Train Loss** | ~2.5 | ~1.0 |
| **Final Val Loss** | ~3.3 | ~1.2 |
| **Best Checkpoint** | Epoch 40-50 | Epoch 30-40 |

**Note**: BPE and character-level losses are not directly comparable due to different vocabulary sizes. BPE predicts from 2048 tokens while character-level predicts from ~88 tokens.

### Equivalent Perplexity

To compare fairly, normalize by average tokens per character:
- BPE: ~3.5 characters per token on average
- Adjusted character-level equivalent loss: ~1.0-1.2

---

## Sample Output

Generated text after training (temperature=1.0, top-k sampling):

```
It was the best of times to be sure, and the old man had not been so
much as a moment's consideration of the subject. The room was small
and dark, with a single candle burning on the mantelpiece, and the
shadows of the evening were beginning to gather in the corners.

"I don't know what you mean," said the stranger, with a slight
tremor in his voice. "I have come here on a matter of business."

The old gentleman looked at him with a curious expression, as if
he were trying to remember something that had happened long ago.
```

The model captures:
- Victorian prose style and vocabulary
- Proper dialogue formatting with quotation marks
- Narrative structure and scene-setting
- Character interactions and emotional undertones

---

## Technical Details

### Attention Implementation

```python
# Scaled dot-product attention with causal masking
def forward(self, x):
    B, T, C = x.shape
    
    # Project to Q, K, V
    q = self.query(x)  # (B, T, head_size)
    k = self.key(x)    # (B, T, head_size)
    v = self.value(x)  # (B, T, head_size)
    
    # Compute attention scores
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
    
    # Apply causal mask (lower triangular)
    attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    attn = self.dropout(attn)
    
    # Apply attention to values
    out = attn @ v
    return out
```

### Generation Strategy

1. Start with seed tokens (e.g., `\n` encoded)
2. Forward pass through model → logits for next token
3. Apply temperature scaling: `logits = logits / temperature`
4. Apply softmax → probability distribution
5. Sample from distribution via `torch.multinomial`
6. Append sampled token to sequence
7. Repeat, using last `context_len` tokens as input

Generation uses `model.eval()` and `torch.no_grad()` for inference efficiency.

---

## Limitations

1. **BPE Training Time**: The standard BPE algorithm iterates over the full corpus for each merge, taking ~3-4 hours for 2048 vocabulary size. This is a one-time cost cached to checkpoint.

2. **Fixed Context Length**: The model can only attend to 256 tokens of context, limiting long-range coherence.

3. **Single Domain**: Trained exclusively on Dickens, limiting generalization to other writing styles.

4. **No Sampling Controls**: Basic temperature-based sampling without nucleus (top-p) or other advanced strategies.

5. **Memory Usage**: Full attention is O(n²) in sequence length, limiting scalability.

---

## Future Work

- [ ] **Optimize BPE Training**: Implement word frequency method to reduce vocabulary construction from hours to minutes
- [ ] **Flash Attention**: Implement memory-efficient attention for longer contexts
- [ ] **Top-p Sampling**: Add nucleus sampling for better generation quality
- [ ] **Multi-corpus Training**: Extend to other Victorian authors
- [ ] **Rotary Positional Embeddings (RoPE)**: Replace learned embeddings for better length generalization
- [ ] **KV-Cache**: Implement key-value caching for faster inference

---

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford, A., et al. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
- Radford, A., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
- Sennrich, R., et al. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE)
- Karpathy, A. [minGPT](https://github.com/karpathy/minGPT) — Minimal GPT implementation reference

---

## License

MIT License

Copyright (c) 2025 FP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
