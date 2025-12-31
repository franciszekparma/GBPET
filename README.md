<p align="center">
  <!-- <img src="images/your-image.png" width="350"/> -->
</p>

<h1 align="center">GBPET</h1>

<p align="center">
  <b>A GPT-style language model with custom BPE tokenization, from scratch.</b><br>
  Trained on Dickens. Writes like it's 1850.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?style=flat-square&logo=pytorch">
  <img src="https://img.shields.io/badge/Parameters-~20M-blue?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

## What is this?

A decoder-only transformer and BPE tokenizer, both implemented from scratch in PyTorch. No HuggingFace, no pretrained anything. Trained on ~15M characters of Charles Dickens novels.

**What's included:**
- Custom Byte Pair Encoding tokenizer
- GPT-2 style transformer (Pre-LayerNorm)
- Full training pipeline with checkpointing
- Bigram baseline for comparison

---

## Model

| | |
|:--|--:|
| Parameters | ~20M |
| Embedding | 512 |
| Heads | 8 |
| Blocks | 8 |
| Context | 256 tokens |
| Vocab | 2,048 (BPE) |

---

## Project Structure

```
GBPET/
├── GBPET/
│   ├── utils.py              # Config
│   ├── data_preparation.py   # BPE tokenizer
│   ├── transformer.py        # Model + training
│   └── bigram.py             # Baseline
├── checkpoints/              # Saved weights
├── data/                     # Dickens corpus
└── samples/                  # Generated text
```

---

## Quick Start

```bash
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET/GBPET
pip install torch numpy tqdm
python transformer.py
```

To generate from a checkpoint, set `ONLY_GENERATE = True` and `TRS_LOAD_CHECKPOINT = True` in `utils.py`.

---

## Training

| | |
|:--|:--|
| Optimizer | AdamW |
| LR Schedule | Warmup + Cosine decay |
| Batch size | 256 |
| Epochs | 128 |
| Hardware | RTX 5090 (BPE) / RTX 4090 (char) |

---

## Results

| | |
|:--|--:|
| Train loss | ~2.5 |
| Val loss | ~3.4 |

```
"You must understand," said Mr. Pickwick, with a grave expression 
upon his countenance, "that circumstances have changed considerably 
since our last meeting."
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [GPT-2](https://openai.com/research/better-language-models) (Radford et al., 2019)
- [BPE](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)

---

<p align="center">
  MIT License · 2025
</p>
