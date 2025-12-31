import torch
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from utils import VOCAB_SIZE, BIGRAM_N_BATCHES, BIGRAM_TRAIN_EPOCHS, BIGRAM_VAL_EPOCHS, LR_BIGRAM, MAX_TOKENS_BIGRAM, BIGRAM_LOAD_CHECKPOINT, BIGRAM_LOAD_PATH, BIGRAM_SAVE_CHECKPOINT, BIGRAM_SAVE_PATH, SET_SEEDS, SEED, DEVICE
from data_preparation import Sample_Batches, train_data, val_data, decode, encode


class Bigram_LM(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
  
  def generate(self, X, max_new_tokens):
    for _ in range(max_new_tokens):
      y_pred = self.forward(X)
      y_pred = y_pred.view(X.shape[0], X.shape[1], -1)
      y_pred = y_pred[:, -1, :]
      
      y_probs = F.softmax(y_pred, dim=-1)
      
      next_token = torch.multinomial(y_probs, num_samples=1)
    
      X = torch.cat((X, next_token), dim=1)
      
    return X
      
  def forward(self, X):
    out = self.table(X)
    out = out.view(out.shape[0] * out.shape[1], out.shape[2])
    
    return out
  
def main():
  SET_SEEDS(SEED)
  
  SB = Sample_Batches(train_data, val_data)

  model = Bigram_LM().to(DEVICE)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BIGRAM)
  loss_fn = nn.CrossEntropyLoss()
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

  start_epoch = 0
  best_val_loss = float('inf')
  
  if BIGRAM_LOAD_CHECKPOINT:
    checkpoint = torch.load(BIGRAM_LOAD_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print("Loaded state dict\n")
  

  for epoch in tqdm(range(start_epoch, BIGRAM_TRAIN_EPOCHS)):
    model.train()
    
    train_losses = []
    val_losses = []
    
    for n_batch in range(BIGRAM_N_BATCHES):
      X, y = SB.get_batch("train")
      X, y = X.to(DEVICE), y.to(DEVICE)
      
      y_hat = model(X)
      
      loss = loss_fn(y_hat, y.view(-1))
      train_losses.append(loss.item())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    model.eval()
    with torch.no_grad():
      for _ in range(BIGRAM_VAL_EPOCHS):
        X, y = SB.get_batch("val")
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        y_hat = model(X)
        
        loss = loss_fn(y_hat, y.view(-1))
        val_losses.append(loss.item())

    avg_train_loss = sum(train_losses)/len(train_losses)
    avg_val_loss = sum(val_losses)/len(val_losses)
    
    scheduler.step(avg_val_loss)
    
    print(f"Epoch: {epoch}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")

    if BIGRAM_SAVE_CHECKPOINT and avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_loss': avg_train_loss
      }, BIGRAM_SAVE_PATH)
      print(f"SAVED NEW CHECKPOINT with Val Loss: {avg_val_loss:.4f}!!!\n")
    
    
  if BIGRAM_SAVE_CHECKPOINT:
    checkpoint = torch.load(BIGRAM_SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])  
    
  start_char = '\n'
  start_idx = encode(start_char)
  start_tokens = torch.tensor(start_idx, dtype=torch.long, device=DEVICE).unsqueeze(0)
  generated = model.generate(start_tokens, max_new_tokens=MAX_TOKENS_BIGRAM)

  print(decode(generated[0].tolist()))
  

if __name__ == '__main__':
  main()