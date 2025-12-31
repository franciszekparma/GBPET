<p align="center">
  <!-- YOUR IMAGE HERE -->
  <!-- <img src="assets/your-image.png" width="400"> -->
  <br><br>
</p>

<h1 align="center">GBPET</h1>
<h3 align="center">GPT with Byte Pair Encoding Tokenizer</h3>

<p align="center">
  A decoder-only transformer language model with custom BPE tokenization,<br>
  implemented entirely from scratch in PyTorch.
</p>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Parameters-~20M-FF6F00?style=flat" alt="Parameters">
  <img src="https://img.shields.io/badge/License-MIT-00C853?style=flat" alt="License">
</p>

<p align="center">
  <b>No HuggingFace</b> · <b>No Pretrained Weights</b> · <b>No External Tokenizers</b>
</p>

<br>

---

<br>

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tokenization](#tokenization)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Generation](#generation)
- [Results](#results)
- [Hardware](#hardware)
- [References](#references)
- [License](#license)

<br>

---

<br>

## Overview

GBPET is a **GPT-style language model** trained on approximately **15 million characters** of Charles Dickens novels. The project demonstrates how to build a modern language model from first principles — every component, from the byte pair encoding tokenizer to the multi-head self-attention mechanism, is implemented from scratch using only PyTorch.

### What's Included

| Component | Description |
|:--|:--|
| **Transformer Model** | Decoder-only architecture with multi-head self-attention, feed-forward networks, and learned positional embeddings |
| **BPE Tokenizer** | Complete implementation of byte pair encoding with vocabulary learning and text encoding/decoding |
| **Training Pipeline** | Full training loop with learning rate scheduling, gradient clipping, and checkpointing |
| **Baseline Model** | Simple bigram model for performance comparison |
| **Pre-trained Checkpoints** | Ready-to-use model weights for text generation |
| **Sample Outputs** | Examples of generated Victorian prose |

### Why From Scratch?

Building a language model from scratch provides deep understanding of:

- How transformers process sequential data
- The mechanics of self-attention and causal masking
- Subword tokenization algorithms
- Training dynamics and optimization strategies

<br>

---

<br>

## Architecture

GBPET follows the **GPT-2 architecture** with Pre-LayerNorm (applying layer normalization before each sub-layer rather than after).

### Model Specifications

<table>
<tr>
<td>

| Parameter | Value |
|:--|--:|
| Total Parameters | ~20M |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Transformer Blocks | 8 |
| Head Dimension | 64 |

</td>
<td>

| Parameter | Value |
|:--|--:|
| Context Length | 256 |
| Vocabulary Size | 2,048 |
| Feed-Forward Dimension | 2,048 |
| Dropout Rate | 0.5 |

</td>
</tr>
</table>

### Design Choices

| Component | Implementation | Rationale |
|:--|:--|:--|
| **Normalization** | Pre-LayerNorm | Improves training stability (GPT-2 style) |
| **Activation** | GELU | Smoother than ReLU, standard in transformers |
| **Positional Encoding** | Learned embeddings | More flexible than sinusoidal |
| **Attention Mask** | Causal (lower triangular) | Prevents attending to future tokens |
| **Output** | Tied with input embeddings | Reduces parameters, improves performance |

### Architecture Diagram

```
Input Token IDs
       |
       v
+------------------+
|  Token Embedding |  (vocab_size=2048, dim=512)
+------------------+
       |
       v
+------------------+
|   + Position     |  (context_len=256, dim=512)
|    Embedding     |
+------------------+
       |
       v
+==========================================+
|           TRANSFORMER BLOCK (x8)         |
|                                          |
|  +------------------------------------+  |
|  |           LayerNorm                |  |
|  +------------------------------------+  |
|                   |                      |
|                   v                      |
|  +------------------------------------+  |
|  |   Multi-Head Self-Attention        |  |
|  |   (8 heads, causal mask)           |  |
|  +------------------------------------+  |
|                   |                      |
|              + Residual                  |
|                   |                      |
|  +------------------------------------+  |
|  |           LayerNorm                |  |
|  +------------------------------------+  |
|                   |                      |
|                   v                      |
|  +------------------------------------+  |
|  |       Feed-Forward Network         |  |
|  |   Linear(512->2048) -> GELU        |  |
|  |   Linear(2048->512)                |  |
|  +------------------------------------+  |
|                   |                      |
|              + Residual                  |
|                                          |
+==========================================+
       |
       v
+------------------+
|    LayerNorm     |
+------------------+
       |
       v
+------------------+
|   Linear Head    |  (dim=512, vocab_size=2048)
+------------------+
       |
       v
   Output Logits
```

<br>

---

<br>

## Tokenization

The project supports two tokenization modes, configurable via the `USE_BYTE_PAIR` flag.

### Byte Pair Encoding (BPE)

A complete implementation of the BPE algorithm from [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909).

#### Algorithm

```
1. INITIALIZE
   - Split text into words at whitespace
   - Treat punctuation (\n, \t, \r, etc.) as separate tokens
   - Add </w> end-of-word marker to each word
   - Create initial vocabulary from unique characters

2. LEARN MERGES
   repeat until vocab_size == 2048:
       - Count all adjacent token pairs in corpus
       - Find most frequent pair (a, b)
       - Merge into new token: ab
       - Add new token to vocabulary
       - Update corpus with merged tokens

3. ENCODE
   - Tokenize text into characters with </w> markers
   - Apply learned merges in order
   - Map tokens to integer IDs

4. DECODE
   - Map IDs back to tokens
   - Join tokens, replace </w> with spaces
```

#### Post-Processing

The `clean_decode()` method fixes spacing artifacts common in BPE outputs:

| Pattern | Before | After |
|:--|:--|:--|
| Contractions | `don ' t` | `don't` |
| Punctuation | `word .` | `word.` |
| Quotes | `" hello "` | `"hello"` |
| Apostrophes | `' s` | `'s` |
| Multiple spaces | `word    word` | `word word` |

#### Vocabulary Statistics

| Metric | Value |
|:--|--:|
| Base characters | ~88 |
| Learned merges | ~1,960 |
| Final vocabulary | 2,048 |
| Special tokens | `</w>`, `</unk>` |

### Character-Level Mode

Set `USE_BYTE_PAIR = False` for simple character tokenization:

- Each unique character becomes a token
- Vocabulary size: ~88 tokens
- No merge learning required
- Useful as baseline comparison

<br>

---

<br>

## Project Structure

```
GBPET/
│
├── GBPET/                          # Source code
│   │
│   ├── utils.py                    # Configuration
│   │   ├── Model hyperparameters   (EMB_DIM, N_HEADS, N_BLOCKS, etc.)
│   │   ├── Training settings       (LR, BATCH_SIZE, EPOCHS, etc.)
│   │   ├── Checkpoint paths        (load/save locations)
│   │   ├── Generation settings     (MAX_TOKENS, START_CHARS)
│   │   └── Mode flags              (USE_BYTE_PAIR, ONLY_GENERATE)
│   │
│   ├── data_preparation.py         # Tokenization & Data
│   │   ├── BytePairEncoding        class
│   │   │   ├── train()             Learn BPE vocabulary
│   │   │   ├── encode()            Text -> Token IDs
│   │   │   ├── decode()            Token IDs -> Text
│   │   │   ├── clean_decode()      Fix spacing artifacts
│   │   │   └── load_checkpoint()   Restore from saved state
│   │   └── Sample_Batches          class
│   │       └── get_batch()         Sample training/validation batches
│   │
│   ├── transformer.py              # Model & Training
│   │   ├── Head                    Single attention head
│   │   ├── MultiHeadAttention      Parallel attention heads + projection
│   │   ├── FeedForward             Position-wise MLP
│   │   ├── Block                   Full transformer block
│   │   ├── Language_Model          Complete model
│   │   │   ├── forward()           Forward pass
│   │   │   └── generate()          Autoregressive generation
│   │   └── main()                  Training loop
│   │
│   └── bigram.py                   # Baseline
│       ├── Bigram_LM               Simple embedding lookup model
│       └── main()                  Training loop
│
├── checkpoints/                    # Saved Models
│   ├── transformer_checkpoint.pt   Trained transformer weights
│   └── bigram_checkpoint.pt        Trained bigram weights
│
├── data/                           # Training Data
│   └── dickens_corpus.txt          ~15M characters of Dickens novels
│
├── samples/                        # Generated Text
│   └── *.txt                       Sample outputs from trained model
│
└── LICENSE                         # MIT License
```

### File Details

#### `utils.py`

Central configuration file. All hyperparameters in one place:

```python
# Model
EMB_DIM = 512
N_HEADS = 8
N_BLOCKS = 8
CONTEXT_LEN = 256
VOCAB_SIZE = 2048  # (or len(VOCAB) for char mode)

# Training
LR_TRS = 6e-4
BATCH_SIZE = 256
TRS_TRAIN_EPOCHS = 128
TRS_N_BATCHES = 256
DP_P = 0.5  # Dropout

# Tokenization
USE_BYTE_PAIR = True
TARGET_VOCAB_SIZE = 2048

# Generation
MAX_TOKENS_TRS = 32768
START_CHARS = '\n'
```

#### `data_preparation.py`

Handles all data processing:

- **BPE Training**: Iteratively learns merge operations from corpus
- **Encoding**: Converts raw text to token IDs
- **Decoding**: Converts token IDs back to readable text
- **Batching**: Samples random context windows for training

#### `transformer.py`

Complete model implementation:

- **Attention**: Scaled dot-product with causal masking
- **Multi-Head**: Parallel attention heads with output projection
- **Feed-Forward**: Two-layer MLP with GELU activation
- **Block**: Pre-norm transformer block with residual connections
- **Model**: Full architecture with embedding, blocks, and output head

<br>

---

<br>

## Getting Started

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

### Installation

```bash
# Clone repository
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET

# Install dependencies
pip install torch numpy tqdm

# For GPU support (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start

```bash
# Navigate to source directory
cd GBPET

# Train the model
python transformer.py

# Or run baseline
python bigram.py
```

<br>

---

<br>

## Training

### Configuration

All training parameters are defined in `utils.py`:

| Parameter | Value | Description |
|:--|--:|:--|
| `LR_TRS` | 6e-4 | Peak learning rate |
| `BATCH_SIZE` | 256 | Samples per batch |
| `TRS_TRAIN_EPOCHS` | 128 | Training epochs |
| `TRS_N_BATCHES` | 256 | Batches per epoch |
| `TRS_VAL_EPOCHS` | 128 | Validation batches |
| `DP_P` | 0.5 | Dropout probability |
| `TRAIN_DATA_FRAC` | 0.8 | Train/val split |

### Learning Rate Schedule

Two-phase schedule using PyTorch's `SequentialLR`:

| Phase | Steps | Schedule |
|:--|--:|:--|
| **Warmup** | 1,638 (5%) | Linear from 0.01x to 1.0x base LR |
| **Decay** | 31,130 (95%) | Cosine annealing to 1e-6 |

```
Learning Rate
     ^
6e-4 |        .--------.
     |       /          \
     |      /            \
     |     /              \
1e-6 |----'                '----------------
     +----+--------+--------+--------------->
          0     1,638    32,768          Steps
          
          Warmup   Cosine Decay
```

### Optimization

| Component | Implementation |
|:--|:--|
| Optimizer | AdamW |
| Gradient Clipping | max_norm = 1.0 |
| Loss Function | CrossEntropyLoss |

### Checkpointing

Automatically saves on validation loss improvement:

```python
checkpoint = {
    # Training state
    'epoch': epoch,
    'train_loss': avg_train_loss,
    'best_val_loss': best_val_loss,
    
    # Model
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    
    # Data
    'encoded_data': data,
    'train_data': train_data,
    'val_data': val_data,
    
    # Architecture (for loading)
    'vocab_size': VOCAB_SIZE,
    'emb_dim': EMB_DIM,
    'context_len': CONTEXT_LEN,
    'n_heads': N_HEADS,
    'n_blocks': N_BLOCKS,
    
    # Tokenizer
    'use_byte_pair': True,
    'bpe_merges': BPE.merges,
    'bpe_st_2_i': BPE.st_2_i,
    'bpe_i_2_st': BPE.i_2_st,
}
```

### Running Training

```bash
cd GBPET
python transformer.py
```

Training progress is displayed via tqdm with loss metrics printed each epoch.

<br>

---

<br>

## Generation

### Configuration Flags

| Flag | Default | Description |
|:--|:--|:--|
| `ONLY_GENERATE` | False | Skip training, only generate |
| `ALWAYS_GENERATE` | True | Generate after training without prompt |
| `TRS_LOAD_CHECKPOINT` | False | Load model from checkpoint |
| `MAX_TOKENS_TRS` | 32,768 | Maximum tokens to generate |
| `START_CHARS` | `'\n'` | Prompt for generation |

### Generate from Checkpoint

```python
# In utils.py
ONLY_GENERATE = True
TRS_LOAD_CHECKPOINT = True
TRS_LOAD_PATH = "checkpoints/transformer_checkpoint.pt"
START_CHARS = "It was the best of times"
```

```bash
python transformer.py
```

### Live Output

The `generate()` method displays text in real-time, clearing the terminal after each token for a streaming effect.

### Sampling

Generation uses **multinomial sampling** from the softmax distribution over the vocabulary. The model generates one token at a time, appending it to the context and repeating until reaching `MAX_TOKENS_TRS`.

<br>

---

<br>

## Results

### Training Metrics

| Metric | Value |
|:--|--:|
| Final Training Loss | 2.5 – 3.0 |
| Final Validation Loss | 3.3 – 3.5 |
| Training Time | ~4-6 hours |

### Sample Output

The model generates coherent Victorian prose with:

- Period-appropriate vocabulary
- Proper dialogue formatting with quotation marks
- Complex sentence structures
- Narrative flow and paragraph transitions

Example (from `samples/`):

```
It was the beginning of a new life for him, and he had no idea 
what to expect. The old house stood silent in the evening fog, 
its windows dark and unwelcoming. He thought of the years that 
had passed since he had last walked these streets, and of the 
faces he had known and loved, now gone forever into the shadows 
of memory.

"You must understand," said Mr. Pickwick, with a grave expression 
upon his countenance, "that circumstances have changed considerably 
since our last meeting. The affairs of which I spoke require 
immediate attention."
```

### Comparison

| Model | Parameters | Val Loss |
|:--|--:|--:|
| Bigram Baseline | ~4M | ~4.5 |
| **GBPET (BPE)** | **~20M** | **~3.4** |
| GBPET (Char) | ~20M | ~3.8 |

<br>

---

<br>

## Hardware

### Training Setup

| Configuration | GPU | Use Case |
|:--|:--|:--|
| **BPE (Token-level)** | NVIDIA RTX 5090 | Primary model |
| **Character-level** | NVIDIA RTX 4090 | Baseline comparison |

*GPUs rented from cloud provider.*

### BPE Vocabulary Training

| Metric | Value |
|:--|--:|
| Runtime | ~3-4 hours |
| Hardware | CPU |
| Frequency | One-time (cached to checkpoint) |

The BPE training iterates over the full corpus at each merge step. Once complete, the vocabulary is saved to the checkpoint and reloaded for subsequent training runs.

### Memory Requirements

| Component | Approximate Size |
|:--|--:|
| Model Parameters | ~80 MB |
| Optimizer State | ~160 MB |
| Encoded Corpus | ~60 MB |
| Full Checkpoint | ~300 MB |

<br>

---

<br>

## References

| Paper | Authors | Year | Link |
|:--|:--|:--|:--|
| Attention Is All You Need | Vaswani et al. | 2017 | [arXiv](https://arxiv.org/abs/1706.03762) |
| Language Models are Unsupervised Multitask Learners | Radford et al. | 2019 | [OpenAI](https://openai.com/research/better-language-models) |
| Neural Machine Translation of Rare Words with Subword Units | Sennrich et al. | 2016 | [arXiv](https://arxiv.org/abs/1508.07909) |

### Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — inspiration for minimal implementation
- Project Gutenberg — public domain Dickens texts

<br>

---

<br>

## License

MIT License

Copyright (c) 2025 Franciszek Parma

See [LICENSE](LICENSE) for details.

<br>

---

<p align="center">
  <sub>Built from scratch with PyTorch</sub>
</p>
