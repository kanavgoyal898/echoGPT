import torch

# hyperparameters
batch_size = 64     # how many independent sequences will we process in parallel?
block_size = 256    # what is the maximum context length for predictions?
max_iters = 2000
eval_intervals = 200
eval_iters = 20
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

fwd_mul = 4
n_heads = 8
n_layers = 8
n_embd = 512
dropout = 0.2