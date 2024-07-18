from train import training_loop
from GPT import GPT, GPTConfig 

modelConfig = GPTConfig(block_size = 32,
                        vocab_size= 64000,
                        n_layer= 2,
                        n_head = 4,
                        n_embd = 64)

model = GPT(modelConfig)
training_loop(model, num_tokens=1000000,B=8, T=32, num_epoch=1, data_path="ArabicGPT/shards")
