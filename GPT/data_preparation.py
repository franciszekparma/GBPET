import torch
import numpy as np

from tqdm.auto import tqdm
import string
from collections import Counter
from copy import deepcopy

from utils import VOCAB, TEXT, USE_BYTE_PAIR, TARGET_VOCAB_SIZE, CONTEXT_LEN, BATCH_SIZE, TRS_LOAD_CHECKPOINT, TRS_LOAD_PATH, LOAD_BPE_ONLY, TRAIN_DATA_FRAC, DEVICE


class Sample_Batches():
  def __init__(self, train_data, val_data):
    
    self.train_data =  train_data
    self.val_data = val_data
  
  def get_batch(self, mode):
    data = self.train_data if mode.upper() == 'TRAIN' else self.val_data
    ix_start = np.random.randint(0, len(data) - CONTEXT_LEN - 1, size=(BATCH_SIZE,))
    
    X = torch.stack([data[i : i + CONTEXT_LEN] for i in ix_start])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix_start])
    
    return X, y
  
  
class BytePairEncoding():
  def __init__(self, text_corpus=TEXT, vocab_size=TARGET_VOCAB_SIZE):
    
    self.text_corpus = text_corpus
    self.vocab_size = vocab_size

    self.corpus = []
    current_word = []
    
    for char in text_corpus:
      if char == ' ':
        if current_word:
          self.corpus.append(current_word + ["</w>"])
          current_word = []
      elif char in string.punctuation or char in '\n\t\r':
        if current_word:
          self.corpus.append(current_word + ["</w>"])
          current_word = []
        
        self.corpus.append([char, "</w>"])
      
      else:
        current_word.append(char)
        
    if current_word:
      self.corpus.append(current_word + ["</w>"])
    
    self.vocab = set()
    
    for element in self.corpus:
      for c in element:
        self.vocab.add(c)
    
    self.vocab.add("</unk>")
        
    
    self.new_n_tokens = vocab_size - len(self.vocab)
    
    self.merges = []
    self.sorted_vocab = None
    self.st_2_i = None
    self.i_2_st = None
  
  def train(self):
    print("Training BPE...")
    for _ in tqdm(range(self.new_n_tokens)):
      potential_tokens = []
      for element in self.corpus:
        for i in range(len(element) - 1):
          potential_tokens.append((element[i], element[i+1]))
      
      if not potential_tokens:
        break
      
      counts = Counter(potential_tokens)
      pair = counts.most_common(1)[0][0]
      new_token = pair[0] + pair[1]
      
      self.merges.append(pair)
      self.vocab.add(new_token)
      
      sub_corpus = deepcopy(self.corpus)
      
      for n, element in enumerate(self.corpus):
        i = 0
        while i < len(sub_corpus[n]) - 1:
          if sub_corpus[n][i] + sub_corpus[n][i+1] == new_token:
            sub_corpus[n][i:] = [new_token] + sub_corpus[n][i+2:]
            
          i += 1
  
      self.corpus = sub_corpus
          
    self.sorted_vocab = sorted(self.vocab)
    self.st_2_i = {s: i for i, s in enumerate(self.sorted_vocab)}
    self.i_2_st = {i: s for i, s in enumerate(self.sorted_vocab)}
    
    print("BPE trained\n")
    
  def encode(self, text):
    tokens = []
    
    for char in text:
      if char == ' ':
        if tokens and tokens[-1] != "</w>":
          tokens.append("</w>")
          
      elif char in string.punctuation or char in '\n\t\r':
        if tokens and tokens[-1] != "</w>":
          tokens.append("</w>")
        tokens.extend([char, "</w>"])
        
      else:
        tokens.append(char)
    
    if tokens and tokens[-1] != "</w>":
      tokens.append("</w>")
    
    for (a, b) in self.merges:
      i = 0
      while i < len(tokens) - 1:
        if tokens[i] == a and tokens[i+1] == b:
          tokens[i: i+2] = [a + b] 
        else:
          i += 1
      
    encoded_text = [self.st_2_i.get(t, self.st_2_i.get('</unk>')) for t in tokens]
    return encoded_text

  def decode(self, ids):
    decoded_text = ''.join([self.i_2_st[idx] for idx in ids])
    decoded_text = decoded_text.replace("</w>", " ").strip()
    return decoded_text
  
  def clean_decode(self, ids):
    text = self.decode(ids)
    
    text = text.replace(" 's", "'s")
    text = text.replace(" 't", "'t")
    text = text.replace(" 'd", "'d")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'm", "'m")
    text = text.replace(" n't", "n't")
    text = text.replace(" 'em", "'em")
    text = text.replace(" ' ", "'")
    text = text.replace(" '", "'")
    text = text.replace("' ", "'")
    text = text.replace(' " ', '"')
    text = text.replace(' "', '"')
    text = text.replace('" ', '"')
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace(" - ", "-")
    text = text.replace(" -- ", "--")
    text = text.replace("-- ", "--")
    text = text.replace(" --", "--")
    text = text.replace(" . . .", "...")
    text = text.replace(" ... ", "...")
    text = text.replace("... ", "...")
    text = text.replace(".", ". ")
    text = text.replace("!", "! ")
    text = text.replace("?", "? ")
    text = text.replace(",", ", ")
    text = text.replace(":", ": ")
    text = text.replace(";", "; ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    text = text.replace(" \n", "\n")
    text = text.replace("\n ", "\n")
    text = text.replace(".'", ". '")
    text = text.replace("?'", "? '")
    text = text.replace(",'", ", '")
    text = text.replace("!'", "! '")
    text = text.replace('."', '. "')
    text = text.replace('?"', '? "')
    text = text.replace(',"', ', "')
    text = text.replace('!"', '! "')
    
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()
  
  def load_checkpoint(self, merges, st_2_i, i_2_st):

    self.merges = merges
    self.st_2_i = st_2_i
    self.i_2_st = i_2_st
    self.sorted_vocab = sorted(st_2_i.keys())
    self.vocab = set(self.sorted_vocab)
    
    print("BPE chekpoint loaded\n")


BPE = None

st_2_i = None
i_2_st = None
encode = None
decode = None


if USE_BYTE_PAIR:
  BPE = BytePairEncoding(TEXT, TARGET_VOCAB_SIZE)
   
  if TRS_LOAD_CHECKPOINT or LOAD_BPE_ONLY:
    checkpoint = torch.load(TRS_LOAD_PATH, map_location=DEVICE, weights_only=False)
    BPE.load_checkpoint(
      checkpoint['bpe_merges'],
      checkpoint['bpe_st_2_i'],
      checkpoint['bpe_i_2_st']
    )
    
    if 'encoded_data' in checkpoint:
      data = checkpoint['encoded_data']
      print("Loaded pre-encoded data\n")
      
    else:
      print("Encoding corpus...")
      data = torch.tensor(BPE.encode(BPE.text_corpus), dtype=torch.long)
      
  else:
    BPE.train()
    data = torch.tensor(BPE.encode(BPE.text_corpus), dtype=torch.long)
    print("Encoding corpus...")
   
  encode = BPE.encode
  decode = BPE.decode
   
    
else:    
  st_2_i = {s : i for i, s in enumerate(VOCAB)}
  i_2_st = {i : s for i, s in enumerate(VOCAB)}

  encode = lambda s: [st_2_i[t] for t in s]
  decode = lambda idc: ''.join([i_2_st[i] for i in idc])

  data = torch.tensor(encode(TEXT), dtype=torch.long)
  print("Encoding corpus...")

f = int(TRAIN_DATA_FRAC * len(data))
train_data, val_data = data[:f], data[f:]
