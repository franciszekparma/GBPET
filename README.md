<p align="center">

```
   ██████╗ ██████╗ ██████╗ ███████╗████████╗
  ██╔════╝ ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
  ██║  ███╗██████╔╝██████╔╝█████╗     ██║   
  ██║   ██║██╔══██╗██╔═══╝ ██╔══╝     ██║   
  ╚██████╔╝██████╔╝██║     ███████╗   ██║   
   ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝   ╚═╝   
```

**GPT with Byte Pair Encoding Tokenizer**

*A transformer language model built entirely from scratch*

</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Parameters-~20M-FF6F00?style=for-the-badge" alt="Parameters">
  <img src="https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge" alt="License">
</p>

<p align="center">
  <code>No HuggingFace</code> · <code>No Pretrained Models</code> · <code>No External Tokenizers</code> · <code>100% From Scratch</code>
</p>

---

<br>

## Overview

GBPET is a **decoder-only transformer language model** trained on **15 million characters** of Charles Dickens novels. 

Every component — from the byte pair encoding tokenizer to the multi-head attention mechanism — is implemented from first principles using only PyTorch.

<br>

---

<br>

## Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      Input Token IDs                                                          ║
║             │                                                                 ║
║             ▼                                                                 ║
║      ┌─────────────────────┐                                                  ║
║      │  Token Embedding    │  nn.Embedding(2048, 512)                         ║
║      └──────────┬──────────┘                                                  ║
║                 │                                                             ║
║                 ▼                                                             ║
║      ┌─────────────────────┐                                                  ║
║      │  Position Embedding │  nn.Embedding(256, 512)   [learned]              ║
║      └──────────┬──────────┘                                                  ║
║                 │                                                             ║
║                 ▼                                                             ║
║      ┌─────────────────────────────────────────────────────────────┐          ║
║      │                  TRANSFORMER BLOCK  (×8)                    │          ║
║      │                                                             │          ║
║      │      ┌─────────────────────────────────────────────┐        │          ║
║      │      │              Layer Norm                     │        │          ║
║      │      └──────────────────┬──────────────────────────┘        │          ║
║      │                         ▼                                   │          ║
║      │      ┌─────────────────────────────────────────────┐        │          ║
║      │      │    Multi-Head Self-Attention  (8 heads)     │        │          ║
║      │      │         scaled dot-product + causal mask    │        │          ║
║      │      └──────────────────┬──────────────────────────┘        │          ║
║      │                         │                                   │          ║
║      │                    + Residual                               │          ║
║      │                         │                                   │          ║
║      │      ┌─────────────────────────────────────────────┐        │          ║
║      │      │              Layer Norm                     │        │          ║
║      │      └──────────────────┬──────────────────────────┘        │          ║
║      │                         ▼                                   │          ║
║      │      ┌─────────────────────────────────────────────┐        │          ║
║      │      │      Feed-Forward Network                   │        │          ║
║      │      │        Linear(512, 2048) → GELU → Linear    │        │          ║
║      │      └──────────────────┬──────────────────────────┘        │          ║
║      │                         │                                   │          ║
║      │                    + Residual                               │          ║
║      │                         │                                   │          ║
║      └─────────────────────────┼───────────────────────────────────┘          ║
║                                │                                              ║
║                                ▼                                              ║
║      ┌─────────────────────────────────────────────────────────────┐          ║
║      │                    Final Layer Norm                         │          ║
║      └──────────────────────────┬──────────────────────────────────┘          ║
║                                 │                                             ║
║                                 ▼                                             ║
║      ┌─────────────────────────────────────────────────────────────┐          ║
║      │                 Linear Head  (512 → 2048)                   │          ║
║      └──────────────────────────┬──────────────────────────────────┘          ║
║                                 │                                             ║
║                                 ▼                                             ║
║                          Output Logits                                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

<br>

<table>
<tr></tr>
<tr>
<td>

### Model Configuration

| Parameter | Value |
|:--|--:|
| Total Parameters | **~20M** |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Transformer Blocks | 8 |
| Context Length | 256 |
| Vocabulary Size | 2048 |
| Feed-Forward Dimension | 2048 |
| Dropout | 0.5 |

</td>
<td>

### Design Choices

| Choice | Implementation |
|:--|:--|
| Normalization | **Pre-Norm** (GPT-2 style) |
| Positional Encoding | **Learned** embeddings |
| Activation | **GELU** |
| Attention Mask | **Causal** (lower triangular) |
| Weight Init | PyTorch defaults |

</td>
</tr>
</table>

<br>

---

<br>

## Tokenization

<table>
<tr></tr>
<tr>
<td width="50%">

### Byte Pair Encoding

Full implementation of the original BPE algorithm.

**Process:**

```
1. Initialize vocabulary with characters + </w>
2. Count all adjacent token pairs in corpus  
3. Merge most frequent pair → new token
4. Repeat until vocab_size = 2048
```

**Example:**

```
Input:   "Hello world"
Tokens:  ['Hel', 'lo</w>', 'world</w>']  
IDs:     [842, 1247, 2019]
```

**Post-processing:**

`clean_decode()` fixes spacing artifacts:
- `don ' t` → `don't`
- `word .` → `word.`
- `" hello "` → `"hello"`

</td>
<td width="50%">

### Character-Level

Baseline tokenization mode.

**Process:**

```
1. Each unique character = one token
2. Vocabulary size ≈ 88
3. Direct character-to-ID mapping
```

**Example:**

```
Input:   "Hello"
Tokens:  ['H', 'e', 'l', 'l', 'o']
IDs:     [39, 58, 65, 65, 72]
```

**Trade-off:**

Simpler but requires longer sequences to represent the same text.

<br>

**Toggle:** Set `USE_BYTE_PAIR = False` in `utils.py`

</td>
</tr>
</table>

<br>

---

<br>

## Project Structure

```
GBPET/
│
├── utils.py ─────────────────────── Configuration
│   │
│   ├── Model:      EMB_DIM, N_HEADS, N_BLOCKS, CONTEXT_LEN
│   ├── Training:   LR, BATCH_SIZE, EPOCHS, DROPOUT
│   ├── Paths:      Checkpoint load/save locations
│   └── Flags:      USE_BYTE_PAIR, ONLY_GENERATE, ALWAYS_GENERATE
│
├── data_preparation.py ──────────── Tokenization & Data Loading
│   │
│   ├── BytePairEncoding
│   │   ├── train()        → Learn vocabulary from corpus
│   │   ├── encode()       → Text to token IDs
│   │   ├── decode()       → Token IDs to text
│   │   └── clean_decode() → Fix spacing artifacts
│   │
│   └── Sample_Batches     → Training/validation batch generator
│
├── transformer.py ───────────────── Model & Training Loop
│   │
│   ├── Head               → Single attention head
│   ├── MultiHeadAttention → Concatenated parallel heads  
│   ├── FeedForward        → Position-wise MLP
│   ├── Block              → Full transformer block
│   ├── Language_Model     → Complete model
│   │
│   └── main()             → Training loop with checkpointing
│
├── bigram.py ────────────────────── Baseline Model
│   │
│   └── Bigram_LM          → Simple embedding lookup baseline
│
└── data/
    └── dickens_corpus.txt ─────── Training corpus (~15M chars)
```

<br>

---

<br>

## Quick Start

### Installation

```bash
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET
pip install torch numpy tqdm
```

### Training

```bash
python transformer.py
```

### Generation Only

```python
# utils.py
ONLY_GENERATE = True
TRS_LOAD_CHECKPOINT = True
```

```bash
python transformer.py
```

### Baseline

```bash
python bigram.py
```

<br>

---

<br>

## Training Configuration

<table>
<tr></tr>
<tr>
<td>

### Optimizer

```
Algorithm ──────── AdamW
Learning Rate ──── 6e-4
Minimum LR ─────── 1e-6
Gradient Clip ──── 1.0
```

</td>
<td>

### Schedule

```
                    Cosine Annealing
                   ╭────────────────╮
Learning      ────╱                  ╲────
Rate         ╱                            ╲
            ╱                              ╲───
           ╱
     Linear Warmup
     
     0 ──── 1,638 ──────────────── 32,768
            (5%)                   steps
```

</td>
<td>

### Data

```
Batch Size ──────── 256
Epochs ─────────── 128
Batches/Epoch ──── 256
Total Steps ────── 32,768
Train/Val ──────── 80/20
```

</td>
</tr>
</table>

<br>

### Checkpoint Contents

```
transformer_checkpoint.pt
│
├── Model State
│   ├── model_state_dict
│   ├── optimizer_state_dict
│   └── scheduler_state_dict
│
├── Data
│   ├── encoded_data      (full corpus as token IDs)
│   ├── train_data
│   └── val_data
│
├── Tokenizer
│   ├── bpe_merges        (learned merge operations)
│   ├── bpe_st_2_i        (token → ID mapping)
│   └── bpe_i_2_st        (ID → token mapping)
│
├── Architecture
│   ├── vocab_size, emb_dim, context_len
│   └── n_heads, n_blocks
│
└── Training State
    ├── epoch, train_loss, best_val_loss
    └── use_byte_pair
```

<br>

---

<br>

## Hardware

<table>
<tr></tr>
<tr>
<td align="center">

### BPE Model

```
┌─────────────────────┐
│                     │
│   NVIDIA RTX 5090   │
│                     │
│   Token-level       │
│   2048 vocabulary   │
│                     │
└─────────────────────┘
```

</td>
<td align="center">

### Character Model

```
┌─────────────────────┐
│                     │
│   NVIDIA RTX 4090   │
│                     │
│   Character-level   │
│   ~88 vocabulary    │
│                     │
└─────────────────────┘
```

</td>
</tr>
</table>

<p align="center"><i>Cloud GPU rental</i></p>

<br>

> **Note:** BPE vocabulary construction takes ~3-4 hours on CPU. This is a one-time cost — the vocabulary is cached to the checkpoint file.

<br>

---

<br>

## Results

<table>
<tr></tr>
<tr>
<td>

### Loss

| Metric | Value |
|:--|--:|
| Training Loss | 2.5 - 3.0 |
| Validation Loss | 3.3 - 3.5 |

</td>
<td>

### Generation Quality

| Aspect | Status |
|:--|:--|
| Victorian prose style | Captured |
| Period vocabulary | Captured |
| Dialogue formatting | Captured |
| Narrative coherence | Captured |

</td>
</tr>
</table>

<br>

### Sample Output

```
It was the beginning of a new life for him, and he had no idea what to expect. 
The old house stood silent in the evening fog, its windows dark and unwelcoming. 
He thought of the years that had passed since he had last walked these streets, 
and of the faces he had known and loved, now gone forever into the shadows of 
memory.

"You must understand," said Mr. Pickwick, with a grave expression upon his 
countenance, "that circumstances have changed considerably since our last 
meeting. The affairs of which I spoke require immediate attention, and I must 
therefore request your assistance in this most delicate matter."
```

<br>

---

<br>

## Features

| | |
|:--|:--|
| **From Scratch** | Every component implemented in pure PyTorch — no external ML libraries |
| **Dual Tokenization** | Switch between BPE and character-level with a single flag |
| **Full Checkpointing** | Save and resume training, or load for generation only |
| **Live Generation** | Real-time text display during inference (clears terminal) |
| **Baseline Model** | Bigram model included for performance comparison |
| **Configurable** | All hyperparameters centralized in `utils.py` |

<br>

---

<br>

## References

| | |
|:--|:--|
| **Attention Is All You Need** | Vaswani et al., 2017 |
| **Language Models are Unsupervised Multitask Learners** | Radford et al., 2019 |
| **Neural Machine Translation of Rare Words with Subword Units** | Sennrich et al., 2016 |

<br>

---

<br>

<p align="center">
  <strong>MIT License</strong>
  <br>
  <sub>Built with PyTorch</sub>
</p>
