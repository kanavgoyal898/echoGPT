import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import *
from parameters import *

torch.manual_seed(2147483647)

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = (int)((0.9)*len(data))
train_data = data[:n]
test_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and test sets
    if (iter+1) % eval_intervals == 0:
        losses = estimate_loss()
        print(f'step {iter+1:5d}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}')

    # sample a batch of data
    x_batch, y_batch = get_batch('train')

    # evaluate the loss
    logits, loss = m(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()