import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32     # how many independent sequences will we process in parallel?
block_size = 8      # what is the maximum context length for predictions?
num_heads = 4
n_embd = 32
max_iters = 50000
eval_intervals = 5000
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

class Head(nn.Module):
    """single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)             # (B, T, head_size)
        q = self.query(x)           # (B, T, head_size)

        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**(-0.5)               # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                            # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)           # (B, T, head_size)
        out = wei @ v               # (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """multiple heads of attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # num_heads heads of head_size dimensional self-attention
        self.sa_head = MultiHeadAttention(num_heads, n_embd//num_heads)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)                                   # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))     # (T, C)
        x = tok_emb + pos_emb                                   # (B, T, C)
        x = self.sa_head(x)                                     # apply one head of self-attention (B, T, head_size)
        logits = self.lm_head(x)                                # (B, T, vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)                                      
            # focus only on the last time step
            logits = logits[:, -1, :]                               # (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)                       # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)      # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1)                # (B, T+k+1)
        return idx                                                  # (B, T+max_new_tokens)
    
model = BigramLanguageModel()
m = model.to(device)

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

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

