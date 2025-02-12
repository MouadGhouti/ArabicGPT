import torch
from torch.nn import functional as F
from ddp import is_ddp

def Generate(model ,tokenizer , example = "السلام عليكم ورحمة الله وبركاته", num_return_sequences=4, max_length=32):
    enc = tokenizer
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = is_ddp()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    model.eval()
    num_return_sequences = num_return_sequences
    max_length = max_length
    tokens = enc.encode(example)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    
    
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
    for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
            
            
            with open('./generation.txt', "a", encoding="utf-8") as f:
                f.write(f"rank {ddp_rank} sample {i}: {decoded} \n")
    print("Written to file")