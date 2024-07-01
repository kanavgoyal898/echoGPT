import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32     # how many independent sequences will we process in parallel?
block_size = 8      # what is the maximum context length for predictions?
max_iters = 10000
eval_intervals = 1000
eval_iters = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------

torch.manual_seed(2147483647)

with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}

encode = lambda s : [stoi[c] for c in s]            # encode: take a string, output a list of integers
decode = lambda l : ''.join([itos[i] for i in l])   # decode: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = (int)((0.9)*len(data))
train_data = data[:n]
test_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)    # (B, T, C)

        # idx and targets are both (B, T) tensors of integers
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for k in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)                                      
            # focus only on the last time step
            logits = logits[:, -1, :]                               # (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)                       # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)      # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1)                # (B, T+k+1)
        return idx                                                  # (B, T+max_new_tokens)
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and test sets
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print(f'step {iter:5d}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}')

    # sample a batch of data
    x_batch, y_batch = get_batch('train')

    # evaluate the loss
    logits, loss = m(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

