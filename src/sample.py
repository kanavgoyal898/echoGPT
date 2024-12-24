import torch

from engine import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the saved model
model_path = 'models/tiny_shakespeare.pt'
model = BigramLanguageModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=False))
model.to(device)
model.eval()

# function to generate text
def generate_text(model, start_string, max_new_tokens):
    # encode the start string
    idx = torch.tensor(encode(start_string), dtype=torch.long).unsqueeze(0).to(device)
    generated_idx = model.generate(idx, max_new_tokens)
    return decode(generated_idx[0].tolist())

# generate text
start_string = 'Et tu, Brute?'
max_new_tokens = int(input('text length: '))
generated_text = generate_text(model, start_string, max_new_tokens)
print(generated_text)