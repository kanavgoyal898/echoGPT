import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64     # how many independent sequences will we process in parallel?
block_size = 256    # what is the maximum context length for predictions?
max_iters = 1
eval_intervals = 1
eval_iters = 1
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fwd_mul = 4
n_heads = 8
n_layers = 8
n_embd = 512
dropout = 0.2
# --------

torch.manual_seed(2147483647)

with open('models/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
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
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)             # (B, T, head_size)
        q = self.query(x)           # (B, T, head_size)

        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**(-0.5)                       # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                    # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)           # (B, T, head_size)
        out = wei @ v               # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, fwd_mul * n_embd),
            nn.ReLU()
        )
        self.proj = nn.Linear(fwd_mul * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.net(x)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # num_heads heads of head_size dimensional self-attention
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)                                   # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))     # (T, C)
        x = tok_emb + pos_emb                                   # (B, T, C)
        x = self.blocks(x)                                      # (B, T, n_embd)
        x = self.ln_f(x)                                        # (B, T, n_embd)
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

param_count = sum([p.nelement() for p in m.parameters()])
print(f'parameter count:', param_count)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and test sets
    if (iter) % eval_intervals == 0:
        losses = estimate_loss()
        print(f'step {iter:5d}: train loss {losses["train"]:.4f}, test loss {losses["test"]:.4f}')

    # sample a batch of data
    x_batch, y_batch = get_batch('train')

    # evaluate the loss
    logits, loss = m(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))