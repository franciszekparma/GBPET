<p align="center">
  <h1 align="center">GBPET</h1>
  <p align="center">
    <strong>GPT + BPE, from scratch, in PyTorch</strong>
  </p>
  <p align="center">
    A ~20M parameter transformer language model trained on Charles Dickens.<br>
    No HuggingFace. No pretrained weights. No external tokenizers.<br>
    <em>Just PyTorch and first principles.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/parameters-~20M-green.svg" alt="Parameters">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

## What is this?

I built a GPT-style language model **completely from scratch** to understand how modern LLMs actually work under the hood:

- **Custom BPE tokenizer** — the same algorithm OpenAI uses, implemented from the original paper
- **Decoder-only transformer** — multi-head attention, feed-forward blocks, the whole thing
- **Trained on 15M characters** of Dickens novels (public domain)
- **Generates coherent Victorian prose** after a few hours of training

No magic. No black boxes. Every line of code written to learn.

## Quick Start

```bash
git clone https://github.com/franciszekparma/GBPET.git
cd GBPET/GBPET
pip install torch numpy tqdm
python transformer.py
```

## Sample Output

> **TODO**: Add real generated samples from the trained model

```bash
# Generate text from checkpoint:
# Set ONLY_GENERATE=True in utils.py, then:
python transformer.py
```

## Model at a Glance

```
┌─────────────────────────────────────┐
│           GBPET Model               │
├─────────────────────────────────────┤
│  Parameters:      ~20M              │
│  Embedding dim:   512               │
│  Attention heads: 8                 │
│  Blocks:          8                 │
│  Context length:  256 tokens        │
│  Vocab size:      2048 (BPE)        │
└─────────────────────────────────────┘
```

**Architecture**: Pre-LayerNorm (GPT-2 style) with learned positional embeddings and GELU activations.

## How the BPE Tokenizer Works

The tokenizer learns subword units by iteratively merging the most frequent character pairs:

```
Iteration 0:    ['T', 'h', 'e', '</w>', 't', 'h', 'e', '</w>']
Iteration 127:  ['The</w>', 'the</w>']
...
Iteration 1960: 2048 tokens learned
```

Starting from ~88 characters, it builds up to 2048 subword tokens — capturing common words, word pieces, and punctuation patterns.

## Training

**Hardware**:
| Mode | GPU |
|------|-----|
| BPE (subword) | RTX 5090 (rented) |
| Character-level | RTX 4090 (rented) |

**Config**:
- AdamW optimizer with linear warmup + cosine decay
- Learning rate: 6e-4 → 1e-6
- Batch size: 256
- Dropout: 0.5
- ~3-4 hours total training time

## Project Structure

```
GBPET/
├── GBPET/
│   ├── utils.py            # All hyperparameters
│   ├── data_preparation.py # BPE tokenizer
│   ├── transformer.py      # Model + training
│   └── bigram.py           # Baseline
├── data/
│   └── dickens_corpus.txt  # Training data
├── checkpoints/
└── samples/
```

## Why Build This?

Because using `from transformers import GPT2` doesn't teach you anything.

Building from scratch means understanding:
- How attention actually computes similarity between tokens
- Why we need positional encodings
- How BPE compression works
- What makes training stable (or unstable)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
- [BPE Paper](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)
- [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy

## License

MIT
