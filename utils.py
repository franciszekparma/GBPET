import torch
import numpy as np
import random

with open('data/dickens_corpus.txt', 'r', encoding='utf-8') as f:
  TEXT = f.read()

USE_BYTE_PAIR = True  # if True, apply BPE merges; if False, use char-level tokenization
TARGET_VOCAB_SIZE = 2048

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 24

def SET_SEEDS(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  
TRS_LOAD_CHECKPOINT = False
TRS_LOAD_PATH = "transformer_checkpoint.pt"
TRS_SAVE_CHECKPOINT = True
TRS_SAVE_PATH = "transformer_checkpoint.pt"

LOAD_BPE_ONLY = False

BIGRAM_LOAD_CHECKPOINT = False
BIGRAM_LOAD_PATH = "bigram_checkpoint.pt"
BIGRAM_SAVE_CHECKPOINT = True
BIGRAM_SAVE_PATH = "bigram_checkpoint.pt"

VOCAB = sorted(list(set(TEXT)))

if USE_BYTE_PAIR:
  VOCAB_SIZE = TARGET_VOCAB_SIZE
else:
  VOCAB_SIZE = len(VOCAB)

BATCH_SIZE = 256
EMB_DIM = 512
CONTEXT_LEN = 256
TRAIN_DATA_FRAC = 0.8

N_HEADS = 8
N_BLOCKS = 8

TRS_TRAIN_EPOCHS = 128
TRS_VAL_EPOCHS = 128
TRS_N_BATCHES = 256
N_STEPS = TRS_TRAIN_EPOCHS * TRS_N_BATCHES
WARMUP_STEPS = N_STEPS // 20
LR_TRS = 6e-4
DP_P = 0.5
MAX_TOKENS_TRS = 32768
START_CHARS = '\n'

BIGRAM_TRAIN_EPOCHS = 64
BIGRAM_VAL_EPOCHS = 64
BIGRAM_N_BATCHES = 32
LR_BIGRAM = 1e-2
MAX_TOKENS_BIGRAM = 256

ONLY_GENERATE = False
ALWAYS_GENERATE = True