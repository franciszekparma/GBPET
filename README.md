# GBPET: GPT with Byte Pair Encoding Tokenizer

A **PyTorch** implementation of a **GPT-style decoder-only transformer** with a custom **Byte Pair Encoding tokenizer**, built entirely **from scratch** for educational purposes. Trained on 15 million characters of Charles Dickens novels.

No HuggingFace. No pretrained weights. No SentencePiece. Just PyTorch and first principles.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technical Background](#technical-background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

---

## Overview

GPT (Generative Pre-trained Transformer) is a decoder-only transformer architecture that generates text autoregressively by predicting the next token given all previous tokens. This implementation demonstrates the core concepts behind modern large language models without relying on pre-built NLP libraries.

This project includes:

- **Custom BPE tokenizer** implementing the original algorithm from Sennrich et al. (2016)
- **Decoder-only transformer** with multi-head self-attention and causal masking
- **Dual tokenization modes** switchable between BPE (subword) and character-level
- **Bigram baseline model** for performance comparison
- **Comprehensive checkpointing** preserving model, optimizer, scheduler, and tokenizer state

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~20M |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Transformer Blocks | 8 |
| Context Length | 256 tokens |
| Vocabulary Size | 2048 (BPE) / ~88 (char) |
| Feed-Forward Dimension | 2048 |
| Dropout | 0.5 |

The model uses **Pre-LayerNorm** (GPT-2 style) where normalization is applied before each sub-layer, **learned positional embeddings**, and **GELU activations**.

### Architecture Diagram

```
Input Token IDs
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Token Embedding + Positional Embedding                      │
│  nn.Embedding(2048, 512) + nn.Embedding(256, 512)            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
╔══════════════════════════════════════════════════════════════╗
║                  TRANSFORMER BLOCK (×8)                      ║
║                                                              ║
║    LayerNorm → Multi-Head Attention (8 heads) → + Residual   ║
║                           │                                  ║
║    LayerNorm → FeedForward (512→2048→512, GELU) → + Residual ║
║                                                              ║
╚══════════════════════════╪═══════════════════════════════════╝
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              LayerNorm → Linear Head (512 → 2048)            │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                     Output Logits
```

---

## Technical Background

### Byte Pair Encoding

BPE is a subword tokenization algorithm that iteratively merges the most frequent adjacent character pairs:

1. Initialize vocabulary with all unique characters + `</w>` end-of-word marker
2. Count frequency of all adjacent token pairs in corpus
3. Merge most frequent pair into a new token
4. Repeat until target vocabulary size (2048) is reached

```
Iteration 0:    ['T', 'h', 'e', '</w>', 'c', 'a', 't', '</w>']
Iteration 50:   ['Th', 'e</w>', 'cat</w>']
Iteration 1960: 2048 learned subword tokens
```

### Causal Self-Attention

Each position can only attend to previous positions, enforced by a lower-triangular mask:

```python
attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
attn = F.softmax(attn, dim=-1)
out = attn @ v
```

### Autoregressive Generation

```python
for _ in range(max_tokens):
    logits = model(context[:, -context_len:])
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    context = torch.cat([context, next_token], dim=1)
```

---

## Project Structure

```
GBPET/
├── GBPET/
│   ├── utils.py            # Hyperparameters and configuration
│   ├── data_preparation.py # BPE tokenizer and data loading
│   ├── transformer.py      # Model architecture and training
│   └── bigram.py           # Baseline model
├── data/
│   └── dickens_corpus.txt  # Training corpus (~15M characters)
├── checkpoints/            # Saved model states
├── samples/                # Generated text samples
├── LICENSE
└── README.md
```

| File | Description |
|------|-------------|
| `utils.py` | All hyperparameters, paths, device config, seed utilities |
| `data_preparation.py` | `BytePairEncoding` class with `train()`, `encode()`, `decode()`, `clean_decode()` |
| `transformer.py` | `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `Language_Model` classes |
| `bigram.py` | Simple next-token baseline for comparison |

---

## Installation

```bash
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET
pip install torch numpy tqdm
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network framework |
| `numpy` | Numerical operations |
| `tqdm` | Progress bars |

---

## Usage

### Training

```bash
cd GBPET
python transformer.py
```

### Generate Text Only

Set in `utils.py`:
```python
ONLY_GENERATE = True
TRS_LOAD_CHECKPOINT = True
```

Then run:
```bash
python transformer.py
```

### Run Baseline

```bash
python bigram.py
```

---

## Configuration

Hyperparameters in `utils.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMB_DIM` | 512 | Embedding dimensions |
| `N_HEADS` | 8 | Number of attention heads |
| `N_BLOCKS` | 8 | Number of transformer blocks |
| `CONTEXT_LEN` | 256 | Maximum sequence length |
| `TARGET_VOCAB_SIZE` | 2048 | BPE vocabulary size |
| `BATCH_SIZE` | 256 | Training batch size |
| `LR_TRS` | 6e-4 | Peak learning rate |
| `DP_P` | 0.5 | Dropout probability |
| `TRS_TRAIN_EPOCHS` | 128 | Number of training epochs |
| `WARMUP_STEPS` | N_STEPS // 20 | Linear warmup steps |
| `USE_BYTE_PAIR` | True | Toggle BPE vs character-level |

---

## Training Process

### Hardware

| Tokenization Mode | GPU | Provider |
|-------------------|-----|----------|
| BPE (subword) | NVIDIA RTX 5090 | RunPod (rented) |
| Character-level | NVIDIA RTX 4090 | RunPod (rented) |

### Learning Rate Schedule

Linear warmup followed by cosine annealing:

```
LR
 │
6e-4 ┤       ╭────────────╮
     │      ╱              ╲
     │     ╱                ╲
1e-6 ┤────╱                  ╲────
     └────┬────────┬─────────┬────→ Steps
          0    ~1.6k      ~33k
        Warmup   Cosine Decay
```

### Training Time

| Component | Duration |
|-----------|----------|
| BPE vocabulary training | ~3-4 hours (one-time, cached) |
| Model training per epoch | ~100-120 seconds |
| Full training run | ~3-4 hours |

---

## Results

### Sample Output
```
"Ah, my prayer," said I.

"Ada, my dear," said Caddy, "you are careful to speak of him?"

"Never, my dear!" I repeated, hurriedly as she made this request,
"don't brighten me any longer, but I don't know."

"You must not say much."

"See where you have said so?"
```

You can generate by:
```bash
# Set ONLY_GENERATE=True in utils.py
python transformer.py
```

### Loss Metrics

| Metric | BPE Model |
|--------|-----------|
| Final Train Loss | ~2.5 |
| Final Val Loss | ~3.3 |

---

## Limitations

| Limitation | Description |
|------------|-------------|
| BPE training time | Standard algorithm is O(n × vocab_size), takes hours |
| Fixed context | 256 tokens maximum, no sliding window |
| Single domain | Trained only on Dickens, limited generalization |
| Basic sampling | Temperature only, no top-p/nucleus sampling |
| O(n²) attention | Memory scales quadratically with sequence length |

---

## Future Improvements

- [ ] Optimize BPE with word frequency method
- [ ] Implement Flash Attention for longer contexts
- [ ] Add top-p (nucleus) sampling
- [ ] Implement KV-cache for faster inference
- [ ] Add Rotary Positional Embeddings (RoPE)
- [ ] Multi-corpus training (other Victorian authors)

---

## References

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Radford et al. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
3. Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
4. Sennrich et al. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
5. Karpathy, A. [nanoGPT]([https://github.com/karpathy/minGPT](https://github.com/karpathy/nanoGPT))

---

## License

This project is licensed under the MIT License.
© franciszekparma
