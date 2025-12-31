# GBPET: GPT with Byte Pair Encoding Tokenizer

A decoder-only transformer language model with a custom Byte Pair Encoding (BPE) tokenizer, built entirely from scratch in PyTorch. Trained on 15 million characters of Charles Dickens novels.

No external ML libraries. No HuggingFace. No pretrained models. No SentencePiece or tiktoken. Everything implemented from first principles.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Tokenization System](#tokenization-system)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Hardware](#hardware)
- [Results](#results)
- [Sample Output](#sample-output)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Future Work](#future-work)
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

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Transformer Blocks | 8 |
| Context Length | 256 tokens |
| Vocabulary Size | 2048 (BPE) / ~88 (char-level) |
| Feed-Forward Dimension | 2048 (4 × emb_dim) |
| Dropout | 0.5 |
| Total Parameters | ~20M |

### Architecture Diagram

```
Input Token IDs
       │
       ▼
┌──────────────────┐
│ Token Embedding  │  nn.Embedding(vocab_size, 512)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Positional     │  nn.Embedding(256, 512)
│   Embedding      │  (learned, not sinusoidal)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Transformer Block (×8)           │
│  ┌────────────────────────────────────┐  │
│  │         LayerNorm (Pre-Norm)       │  │
│  └────────────────┬───────────────────┘  │
│                   │                      │
│                   ▼                      │
│  ┌────────────────────────────────────┐  │
│  │    Multi-Head Self-Attention       │  │
│  │    (8 heads, causal masking)       │  │
│  └────────────────┬───────────────────┘  │
│                   │ + Residual           │
│                   ▼                      │
│  ┌────────────────────────────────────┐  │
│  │         LayerNorm (Pre-Norm)       │  │
│  └────────────────┬───────────────────┘  │
│                   │                      │
│                   ▼                      │
│  ┌────────────────────────────────────┐  │
│  │    Feed-Forward Network            │  │
│  │    (512 → 2048 → 512, GELU)        │  │
│  └────────────────┬───────────────────┘  │
│                   │ + Residual           │
│                   ▼                      │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Final LayerNorm │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Linear Head     │  (512 → vocab_size)
└────────┬─────────┘
         │
         ▼
   Output Logits
```

The architecture follows the **Pre-Norm** convention (GPT-2 style): LayerNorm is applied *before* each sub-layer (attention, FFN), rather than after. This improves training stability.

---

## Tokenization System

The project supports two tokenization modes, controlled by the `USE_BYTE_PAIR` flag in `utils.py`:

### Byte Pair Encoding (BPE)

The BPE tokenizer is implemented entirely from scratch, following the original algorithm.

**Algorithm:**

1. **Corpus Preprocessing**: Text is split into words at spaces. Punctuation and special characters (`\n`, `\t`, `\r`) are treated as separate tokens. Each word ends with a special `</w>` end-of-word marker.

2. **Vocabulary Initialization**: The initial vocabulary consists of all unique characters in the corpus plus `</unk>` for unknown tokens (approximately 88 base tokens).

3. **Merge Learning**: The algorithm iteratively:
   - Counts all adjacent token pairs across the corpus
   - Finds the most frequent pair
   - Merges them into a new token
   - Updates the corpus with the merged token
   - Repeats until `TARGET_VOCAB_SIZE` (2048) is reached

4. **Encoding**: To encode text, it first tokenizes into characters with `</w>` markers, then applies all learned merges in order. Each token is mapped to an integer ID via the `st_2_i` dictionary.

5. **Decoding**: Integer IDs are mapped back to tokens via the `i_2_st` dictionary, concatenated, and `</w>` is replaced with spaces.

**Post-Processing (`clean_decode`):**

The `clean_decode()` method handles spacing artifacts after decoding:
- Fixes contractions: `don ' t` → `don't`, `I ' m` → `I'm`
- Handles punctuation spacing: removes spaces before periods, commas, etc.
- Fixes quotation marks and apostrophes
- Normalizes multiple spaces
- Preserves literary conventions like em-dashes

### Character-Level Tokenization

When `USE_BYTE_PAIR = False`, the model uses simple character-level tokenization:
- Each unique character in the corpus becomes a token
- Vocabulary size is approximately 88 tokens
- Simpler but requires longer sequences to represent the same text

---

## Project Structure

```
GBPET/
├── utils.py              # Hyperparameters and configuration
├── data_preparation.py   # BPE tokenizer class and data loading
├── transformer.py        # Transformer model and training loop
├── bigram.py             # Bigram baseline model
├── transformer_checkpoint.pt   # Model checkpoint (generated)
├── bigram_checkpoint.pt        # Bigram checkpoint (generated)
└── data/
    └── dickens_corpus.txt      # Training corpus (~15M characters)
```

### File Descriptions

| File | Description |
|------|-------------|
| `utils.py` | Central configuration file. Defines all hyperparameters: `EMB_DIM`, `N_HEADS`, `N_BLOCKS`, `CONTEXT_LEN`, `VOCAB_SIZE`, learning rate, batch size, dropout rate, checkpoint paths, and random seeds. Toggle `USE_BYTE_PAIR` to switch tokenization modes. |
| `data_preparation.py` | Contains the `BytePairEncoding` class with `train()`, `encode()`, `decode()`, and `clean_decode()` methods. Also implements `Sample_Batches` for train/val data loading. Handles both BPE and character-level tokenization based on config. |
| `transformer.py` | Implements `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `Language_Model` classes. Contains the training loop with warmup + cosine annealing LR schedule, gradient clipping, and checkpointing. Includes real-time text generation display. |
| `bigram.py` | Simple bigram language model (`Bigram_LM`) for baseline comparison. Uses an embedding table where each token directly predicts the next token distribution. |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA support recommended)
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

For GPU training (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Configuration

Edit `utils.py` to configure the model:

```python
# Tokenization mode
USE_BYTE_PAIR = True          # True for BPE, False for character-level

# Checkpoint settings
TRS_LOAD_CHECKPOINT = False   # Load existing checkpoint
TRS_SAVE_CHECKPOINT = True    # Save checkpoints during training
LOAD_BPE_ONLY = False         # Load only BPE vocab (train fresh model)

# Generation settings
ONLY_GENERATE = False         # Skip training, only generate
ALWAYS_GENERATE = True        # Generate after training without prompt
```

### Training the Transformer

```bash
python transformer.py
```

This will:
1. Load the Dickens corpus from `data/dickens_corpus.txt`
2. Train the BPE tokenizer (or load from checkpoint if available)
3. Encode the corpus
4. Train the transformer model with warmup + cosine annealing
5. Save checkpoints when validation loss improves
6. Generate sample text after training

### Training the Bigram Baseline

```bash
python bigram.py
```

### Generation Only

Set `ONLY_GENERATE = True` in `utils.py`, then:
```bash
python transformer.py
```

### Loading from Checkpoint

```python
import torch
from data_preparation import BytePairEncoding
from transformer import Language_Model
from utils import DEVICE

# Load checkpoint
checkpoint = torch.load('transformer_checkpoint.pt', map_location=DEVICE)

# Restore BPE tokenizer
BPE = BytePairEncoding.__new__(BytePairEncoding)
BPE.load_checkpoint(
    checkpoint['bpe_merges'],
    checkpoint['bpe_st_2_i'],
    checkpoint['bpe_i_2_st']
)

# Load model
model = Language_Model(
    vocab_size=checkpoint['vocab_size'],
    emb_dim=checkpoint['emb_dim'],
    context_len=checkpoint['context_len'],
    n_heads=checkpoint['n_heads'],
    n_blocks=checkpoint['n_blocks']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Generate
prompt = "It was the best of times"
tokens = torch.tensor([BPE.encode(prompt)], dtype=torch.long, device=DEVICE)
model.generate(tokens, max_new_tokens=500)
```

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 6e-4 (peak) |
| LR Schedule | Linear Warmup → Cosine Annealing |
| Warmup Steps | 5% of total steps (1,638 steps) |
| Min LR | 1e-6 |
| Batch Size | 256 |
| Gradient Clipping | max_norm=1.0 |
| Dropout | 0.5 |
| Training Epochs | 128 |
| Batches per Epoch | 256 |
| Total Steps | 32,768 |
| Train/Val Split | 80% / 20% |

### Learning Rate Schedule

The training uses a two-phase learning rate schedule via `SequentialLR`:

1. **Linear Warmup** (`LinearLR`): LR increases from 0.01 × base_lr to base_lr over the first 1,638 steps (5% of training)
2. **Cosine Annealing** (`CosineAnnealingLR`): LR decreases from base_lr to 1e-6 following a cosine curve

```python
warmup_lrs = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_STEPS)
cosine_lrs = CosineAnnealingLR(optimizer, T_max=N_STEPS-WARMUP_STEPS, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup_lrs, cosine_lrs], milestones=[WARMUP_STEPS])
```

### Checkpoint Contents

The checkpoint saves:

| Key | Description |
|-----|-------------|
| `model_state_dict` | Model weights |
| `optimizer_state_dict` | Optimizer state (momentum, etc.) |
| `scheduler_state_dict` | LR scheduler state |
| `encoded_data` | Pre-encoded training corpus |
| `train_data` | Training split tensor |
| `val_data` | Validation split tensor |
| `epoch` | Current training epoch |
| `best_val_loss` | Best validation loss achieved |
| `train_loss` | Last training loss |
| `vocab_size`, `emb_dim`, `context_len`, `n_heads`, `n_blocks` | Architecture config |
| `use_byte_pair` | Tokenization mode flag |
| `bpe_merges`, `bpe_st_2_i`, `bpe_i_2_st` | BPE vocabulary (if BPE mode) |
| `char_st_2_i`, `char_i_2_st` | Character vocabulary (if char mode) |

---

## Hardware

The model was trained on rented cloud GPUs:

| Tokenization Mode | GPU | Notes |
|-------------------|-----|-------|
| **BPE (Token-level)** | NVIDIA RTX 5090 | Subword tokenization with 2048 vocab |
| **Character-level** | NVIDIA RTX 4090 | ~88 character vocabulary |

BPE training (vocabulary construction) takes approximately 3-4 hours on CPU as it iterates over the full corpus at each merge step. The encoded corpus is cached to checkpoint, so this only runs once.

---

## Results

### Loss Metrics

| Metric | Value |
|--------|-------|
| Training Loss | ~2.5 - 3.0 |
| Validation Loss | ~3.3 - 3.5 |

### Bigram Baseline

The bigram model provides a baseline for comparison. It uses a simple embedding table where each token directly predicts the probability distribution over the next token, without any context beyond the current token.

---

## Sample Output

After training, the model generates coherent Dickens-style prose. The `generate()` method clears the terminal and displays text in real-time as tokens are generated:

```
It was the beginning of a new life for him, and he had no idea what to 
expect. The old house stood silent in the evening fog, its windows dark 
and unwelcoming. He thought of the years that had passed since he had 
last walked these streets, and of the faces he had known and loved, 
now gone forever into the shadows of memory.

"You must understand," said Mr. Pickwick, with a grave expression upon 
his countenance, "that circumstances have changed considerably since 
our last meeting. The affairs of which I spoke require immediate 
attention, and I must therefore request your assistance in this most 
delicate matter."
```

The model captures:
- Victorian prose style and sentence structure
- Period-appropriate vocabulary
- Dialogue formatting conventions
- Paragraph transitions and narrative flow

---

## Technical Details

### Attention Mechanism

Each attention head computes scaled dot-product attention with causal masking:

```python
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, dp_p):
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN)))
        self.dropout = nn.Dropout(dp_p)
        
    def forward(self, X):
        k, q, v = self.key(X), self.query(X), self.value(X)
        
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei + torch.where(self.tril[:X.shape[1], :X.shape[1]] == 0, float('-inf'), 0.0)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        return wei @ v
```

### Multi-Head Attention

Multiple heads are computed in parallel and concatenated:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, emb_dim, dp_p):
        self.heads = nn.ModuleList([Head(emb_dim, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dp_p)
        
    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
```

### Feed-Forward Network

Position-wise feed-forward network with GELU activation:

```python
class FeedForward(nn.Module):
    def __init__(self, emb_dim, dp_p):
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dp_p),
            nn.Linear(emb_dim * 4, emb_dim)
        )
```

### Transformer Block (Pre-Norm)

```python
class Block(nn.Module):
    def forward(self, X):
        out = X + self.sa_heads(self.ln1(X))   # Pre-norm: LN before attention
        out = out + self.ffd(self.ln2(out))    # Pre-norm: LN before FFN
        return out
```

---

## Limitations

1. **BPE Training Time**: The standard BPE algorithm iterates over the full corpus at each merge step, requiring 3-4 hours for vocabulary construction. This only runs once and is cached to checkpoint.

2. **Vocabulary Size**: With 2048 tokens, the model cannot represent highly specialized or rare words as single tokens. Increasing vocabulary size would improve this but requires more training data.

3. **Context Length**: The 256-token context limits the model's ability to maintain coherence over very long passages. Extending context requires quadratically more memory for attention.

4. **Training Data**: 15M characters is relatively small for language models. Larger corpora would improve generation quality and diversity.

5. **Single Author**: Training exclusively on Dickens creates a model specialized for Victorian prose. Generalization to other styles is limited.

6. **Sampling Strategy**: Generation uses simple multinomial sampling. Beam search or nucleus sampling could improve output quality.

7. **No KV-Cache**: Generation recomputes attention for all previous tokens at each step. Implementing KV-cache would significantly speed up inference.

---

## Future Work

- [ ] Optimize BPE training using word frequency method (reduce from ~3.5h to ~5min)
- [ ] Implement KV-cache for faster generation
- [ ] Add nucleus (top-p) and top-k sampling options
- [ ] Implement temperature scaling for generation diversity control
- [ ] Add beam search decoding
- [ ] Experiment with larger vocabulary sizes (4096, 8192)
- [ ] Implement Flash Attention for memory efficiency
- [ ] Extend context length with RoPE or ALiBi positional encodings
- [ ] Train on larger, multi-author corpus
- [ ] Add perplexity evaluation on held-out test set
- [ ] Implement mixed-precision training (fp16/bf16)

---

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2016)
- Karpathy, "nanoGPT" - minimal GPT implementation

---

## License

MIT License

Copyright (c) 2024-2025 Franciszek Parma

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
