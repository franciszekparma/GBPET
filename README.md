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

## Overview

GBPET is a **GPT-style language model** trained on ~15 million characters of Charles Dickens novels. Every component — from the BPE tokenizer to multi-head self-attention — is implemented from scratch using only PyTorch.

<br>

---

<br>

## Architecture

<table>
<tr>
<td>

| Parameter | Value |
|:--|--:|
| Parameters | ~20M |
| Embedding dim | 512 |
| Attention heads | 8 |
| Blocks | 8 |

</td>
<td>

| Parameter | Value |
|:--|--:|
| Context length | 256 |
| Vocabulary | 2,048 |
| FFN dim | 2,048 |
| Dropout | 0.5 |

</td>
</tr>
</table>

Uses **Pre-LayerNorm**, **GELU** activation, and **learned positional embeddings**.

```
Input IDs
    │
    ▼
┌────────────────┐
│ Token + Position Embedding
└────────────────┘
    │
    ▼
┌────────────────────────────────┐
│     TRANSFORMER BLOCK (x8)     │
│                                │
│  LayerNorm → Multi-Head Attn   │
│         + Residual             │
│  LayerNorm → Feed-Forward      │
│         + Residual             │
└────────────────────────────────┘
    │
    ▼
┌────────────────┐
│ LayerNorm → Linear Head → Logits
└────────────────┘
```

<br>

---

<br>

## Tokenization

### BPE (Byte Pair Encoding)

Full implementation of the original algorithm ([Sennrich et al., 2016](https://arxiv.org/abs/1508.07909)):

1. Initialize vocabulary with characters + `</w>` end-of-word markers
2. Count all adjacent token pairs in corpus
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reaches 2,048

The `clean_decode()` method fixes spacing artifacts (contractions, punctuation, quotes).

Set `USE_BYTE_PAIR = False` in `utils.py` for character-level tokenization.

<br>

---

<br>

## Project Structure

```
GBPET/
├── GBPET/
│   ├── utils.py              # Configuration and hyperparameters
│   ├── data_preparation.py   # BPE tokenizer and data loading
│   ├── transformer.py        # Model architecture and training
│   └── bigram.py             # Baseline model
│
├── checkpoints/              # Saved model weights
├── data/                     # Training corpus (Dickens)
├── samples/                  # Generated text samples
└── LICENSE
```

<br>

---

<br>

## Getting Started

```bash
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET
pip install torch numpy tqdm
```

**Train**
```bash
cd GBPET
python transformer.py
```

**Generate only** (set in `utils.py`)
```python
ONLY_GENERATE = True
TRS_LOAD_CHECKPOINT = True
```

<br>

---

<br>

## Training

| Parameter | Value |
|:--|--:|
| Optimizer | AdamW |
| Learning rate | 6e-4 |
| Schedule | Linear warmup (5%) → Cosine decay |
| Batch size | 256 |
| Epochs | 128 |
| Gradient clipping | 1.0 |

Checkpoints automatically save model, optimizer, scheduler, tokenizer, and encoded corpus.

<br>

---

<br>

## Results

| Metric | Value |
|:--|--:|
| Training loss | 2.5 – 3.0 |
| Validation loss | 3.3 – 3.5 |

Sample outputs available in `samples/`. The model generates coherent Victorian prose with period-appropriate vocabulary and dialogue formatting.

<br>

---

<br>

## Hardware

| Mode | GPU |
|:--|:--|
| BPE (token-level) | NVIDIA RTX 5090 |
| Character-level | NVIDIA RTX 4090 |

BPE vocabulary training: ~3-4 hours on CPU (one-time, cached to checkpoint).

<br>

---

<br>

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019)
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016)

<br>

---

<br>

## License

MIT — see [LICENSE](LICENSE)

<br>
