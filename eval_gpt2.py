from train_gpt2 import GPT, GPTConfig
import torch
from transformers import AutoTokenizer

model = GPT(GPTConfig(block_size=1024, vocab_size=128010, n_layer=12, n_head=12, n_embd=768))
model = torch.compile(model)

model.load_state_dict(torch.load("ghc_final.bin"))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print("input:")

tokens = tokenizer.encode(input())

xg = torch.tensor(tokens, dtype=torch.long)[None, ...]

print(tokenizer.decode(model.generate(xg, 32, temperature=1.0, top_k=40)[0].tolist()))