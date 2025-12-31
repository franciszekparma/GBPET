import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_

import os
from tqdm.auto import tqdm

from utils import VOCAB_SIZE, EMB_DIM, N_HEADS, N_BLOCKS, CONTEXT_LEN, TRS_TRAIN_EPOCHS, TRS_VAL_EPOCHS, TRS_N_BATCHES, N_STEPS, WARMUP_STEPS, LR_TRS, DP_P, MAX_TOKENS_TRS, START_CHARS, TRS_LOAD_CHECKPOINT, TRS_LOAD_PATH, TRS_SAVE_CHECKPOINT, TRS_SAVE_PATH, SET_SEEDS, SEED, DEVICE, USE_BYTE_PAIR, ONLY_GENERATE, ALWAYS_GENERATE
from data_preparation import Sample_Batches, train_data, val_data, data, BPE, st_2_i, i_2_st, decode, encode


class Head(nn.Module):
  def __init__(self, emb_dim=EMB_DIM, head_size=EMB_DIM//N_HEADS, dp_p=DP_P):
    super().__init__()
    self.head_size = head_size
    
    self.key = nn.Linear(emb_dim, head_size, bias=False)
    self.query = nn.Linear(emb_dim, head_size, bias=False)
    self.value = nn.Linear(emb_dim, head_size, bias=False)
    
    self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_LEN,  CONTEXT_LEN)))
    
    self.dropout = nn.Dropout(dp_p)
    
  def forward(self, X):
    k = self.key(X)
    q = self.query(X)
    v = self.value(X)
    
    wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
    wei = wei + torch.where(self.tril[:X.shape[1], :X.shape[1]] == 0, float('-inf'), 0.0)
    wei = F.softmax(wei, dim=-1) 
    wei = self.dropout(wei)
    out = wei @ v
    
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads=N_HEADS, head_size=EMB_DIM//N_HEADS, emb_dim=EMB_DIM, dp_p=DP_P):
    super().__init__()
    
    self.heads = nn.ModuleList([Head(emb_dim, head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(emb_dim, emb_dim)
    self.dropout = nn.Dropout(dp_p)
    
  def forward(self, X):
    out = torch.cat([h(X) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out


class FeedForward(nn.Module):
  def __init__(self, emb_dim=EMB_DIM, dp_p=DP_P):
    super().__init__()
    
    self.mlp = nn.Sequential(
      nn.Linear(emb_dim, emb_dim*4),
      nn.GELU(),
      nn.Dropout(dp_p),
      nn.Linear(emb_dim*4, emb_dim)
    )
    
  def forward(self, X):
    return self.mlp(X)


class Block(nn.Module):
  def __init__(self, emb_dim=EMB_DIM, n_heads=N_HEADS):
    super().__init__()
    head_size = emb_dim // n_heads
    
    self.sa_heads = MultiHeadAttention(n_heads, head_size, emb_dim)
    self.ffd = FeedForward(emb_dim)
    
    self.ln1 = nn.LayerNorm(emb_dim)
    self.ln2 = nn.LayerNorm(emb_dim)
  
  def forward(self, X):
    out = X + self.sa_heads(self.ln1(X))
    out = out + self.ffd(self.ln2(out))
    
    return out
    

class Language_Model(nn.Module):
  def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, context_len=CONTEXT_LEN, head_size=EMB_DIM//N_HEADS, n_heads=N_HEADS, n_blocks=N_BLOCKS):
    super().__init__()
    self.context_len = CONTEXT_LEN
    
    self.emb_table = nn.Embedding(vocab_size, emb_dim)
    self.positional_encoding = nn.Embedding(context_len, emb_dim)
    self.blocks = nn.Sequential(*[Block(emb_dim, n_heads) for _ in range(n_blocks)])
    self.ln = nn.LayerNorm(emb_dim)
    self.lm_head = nn.Linear(emb_dim, vocab_size)
    
  def generate(self, ids, max_new_tokens):
    with torch.no_grad():
      for _ in range(max_new_tokens):
        context = ids[:, -self.context_len:]
        
        y_pred = self.forward(context)
        y_pred = y_pred.view(context.shape[0], context.shape[1], -1)
        y_pred = y_pred[:, -1, :]
        
        y_probs = F.softmax(y_pred, dim=-1)
        
        next_token = torch.multinomial(y_probs, num_samples=1)
      
        ids = torch.cat((ids, next_token), dim=1)
         
        os.system('cls' if os.name == 'nt' else 'clear')
        if USE_BYTE_PAIR:
          print(BPE.clean_decode(ids[0].tolist()))
        else:
          print(decode(ids[0].tolist()))


  def forward(self, ids):
    tok_emb = self.emb_table(ids)
    pos_enc = self.positional_encoding(torch.arange(ids.shape[1], device=DEVICE))
    
    out = tok_emb + pos_enc
    out = self.ln(self.blocks(out))
    out = self.lm_head(out)
    
    out = out.view(out.shape[0] * out.shape[1], -1)
    
    return out
  
  
def main():
  SET_SEEDS(SEED)
  
  SB = Sample_Batches(train_data, val_data)

  model = Language_Model().to(DEVICE)

  optimizer = torch.optim.AdamW(model.parameters(), lr=LR_TRS)
  loss_fn = nn.CrossEntropyLoss()
  warmup_lrs = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_STEPS)
  cosine_lrs = CosineAnnealingLR(optimizer, T_max=N_STEPS-WARMUP_STEPS, eta_min=1e-6)
  scheduler = SequentialLR(optimizer, [warmup_lrs, cosine_lrs], milestones=[WARMUP_STEPS])

  start_epoch = 0
  best_val_loss = float('inf')
  
  if TRS_LOAD_CHECKPOINT:
    checkpoint = torch.load(TRS_LOAD_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print("Loaded state dict\n")


  if not ONLY_GENERATE:
    
    for epoch in tqdm(range(start_epoch, TRS_TRAIN_EPOCHS)):
      model.train()
      
      train_losses = []
      val_losses = []
      
      for n_batch in range(TRS_N_BATCHES):
        X, y = SB.get_batch('train')
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        y_hat = model(X)
        
        loss = loss_fn(y_hat, y.view(-1))
        train_losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
      
      model.eval()
      with torch.no_grad():
        for _ in range(TRS_VAL_EPOCHS):
          X, y = SB.get_batch("val")
          X, y = X.to(DEVICE), y.to(DEVICE)
          
          y_hat = model(X)
          
          loss = loss_fn(y_hat, y.view(-1))
          val_losses.append(loss.item())
      
      
      avg_train_loss = sum(train_losses)/len(train_losses)
      avg_val_loss = sum(val_losses)/len(val_losses)

      print(f"Epoch: {epoch}")
      print(f"Train Loss: {avg_train_loss:.4f}")
      print(f"Val Loss: {avg_val_loss:.4f}")
      
      if TRS_SAVE_CHECKPOINT and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        save_dict = {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'encoded_data': data,
          'train_data': train_data,
          'val_data': val_data,
          'best_val_loss': best_val_loss,
          'train_loss': avg_train_loss,
          'vocab_size': VOCAB_SIZE,
          'emb_dim': EMB_DIM,
          'context_len': CONTEXT_LEN,
          'n_heads': N_HEADS,
          'n_blocks': N_BLOCKS,
        }
        if USE_BYTE_PAIR:
          save_dict['use_byte_pair'] = True
          save_dict['bpe_merges'] = BPE.merges
          save_dict['bpe_st_2_i'] = BPE.st_2_i
          save_dict['bpe_i_2_st'] = BPE.i_2_st
        else:
          save_dict['use_byte_pair'] = False
          save_dict['char_st_2_i'] = st_2_i
          save_dict['char_i_2_st'] = i_2_st
          
        torch.save(save_dict, TRS_SAVE_PATH)
        
        print(f"SAVED NEW CHECKPOINT with Val Loss: {avg_val_loss:.4f}!!!\n")


    if TRS_SAVE_CHECKPOINT:
      checkpoint = torch.load(TRS_SAVE_PATH, map_location=DEVICE, weights_only=True)
      model.load_state_dict(checkpoint['model_state_dict'])
  
  
  if ALWAYS_GENERATE:
    
    if USE_BYTE_PAIR:
      start_idx = BPE.encode(START_CHARS)
      start_tokens = torch.tensor(start_idx, dtype=torch.long, device=DEVICE).unsqueeze(0)
      model.eval()
      model.generate(start_tokens, max_new_tokens=MAX_TOKENS_TRS)
      
    else:
      start_idx = encode(START_CHARS)
      start_tokens = torch.tensor(start_idx, dtype=torch.long, device=DEVICE).unsqueeze(0)
      model.eval()
      model.generate(start_tokens, max_new_tokens=MAX_TOKENS_TRS)
    
  else:
    d = input("\nSTART GENERATING [Y/N]: ")
    if d.upper() == "Y":
      
      if USE_BYTE_PAIR:
        start_idx = BPE.encode(START_CHARS)
        start_tokens = torch.tensor(start_idx, dtype=torch.long, device=DEVICE).unsqueeze(0)
        model.eval()
        model.generate(start_tokens, max_new_tokens=MAX_TOKENS_TRS)
      
      else:
        start_idx = encode(START_CHARS)
        start_tokens = torch.tensor(start_idx, dtype=torch.long, device=DEVICE).unsqueeze(0)
        model.eval()
        model.generate(start_tokens, max_new_tokens=MAX_TOKENS_TRS)
  

if __name__ == '__main__':
  main()