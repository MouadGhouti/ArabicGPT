from train import training_loop
from GPT import GPT, GPTConfig 
from ddp import is_master_process
from transformers import PreTrainedTokenizerFast
from generate import Generate


#Create a model and train it for 1 epoch
# modelConfig = GPTConfig(block_size = 1024,
#                         vocab_size= 64000,
#                         n_layer= 12,
#                         n_head = 12,
#                         n_embd = 768,
#                         master_process= True)# use is_master_process()

# model = GPT(modelConfig)
# training_loop(model, num_tokens=1000000,B=8, T=32, num_epoch=1, data_path="ArabicGPT/shards")
#______________________________________________________________________________________________


#Load a model and generate text
pretrained_model = GPT.from_pretrained(model_path='ArabicGPT')
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"ArabicGPT/tokenizers/TBPE_tokenizer_64.0K.json")
Generate(pretrained_model, tokenizer)

