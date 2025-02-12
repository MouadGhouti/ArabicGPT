from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ddp import is_ddp
from GPT import GPT, GPTConfig
from dataloader import DataLoaderLite
import torch
import os
import time
import math
from transformers import PreTrainedTokenizerFast


torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


def set_batch_size(B1=32, T1=1024, ddp_world_size=1):
    total_batch_size = B1 * T1 * ddp_world_size
    assert total_batch_size % (B1 * T1 * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B1 * T1 * ddp_world_size)
    return B1 ,T1, total_batch_size, grad_accum_steps

def get_num_steps(total_batch_size, total_number_of_tokens=4e9):
    return total_number_of_tokens//total_batch_size


def training_loop(passed_model, num_tokens, B=32, T=1024, num_epoch=1, data_path="ArabicGPT/shards"):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = is_ddp()
    
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    B ,T, total_batch_size, grad_accum_steps = set_batch_size(B1=B, T1=T, ddp_world_size=ddp_world_size)#CHANGE BATCH SIZE HERE

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    num_epochs = num_epoch
    num_steps = get_num_steps(total_batch_size, total_number_of_tokens=num_tokens)#CHANGE TOTAL NUMBER OF TOKENS HERE


    max_steps = num_steps * num_epochs #if data is 10B tokens and batch size 0.5M tokens (10B in 2 hours on 8 GPUs)

    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"Number of steps per epoch: {num_steps}")
        print(f"Number of epochs: {num_epochs}")
        print("Number of GPUs:",ddp_world_size)



    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process, data_path=data_path)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process,  data_path=data_path)




    # create model
    model = passed_model
    model.to(device)
    use_compile = True 
    if use_compile:
        try:
            model = torch.compile(model)
        except:
            print("torch.compile failed. Switching to torch.compile mode disabled.")
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model



    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass


    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 10
                for _ in range(val_loss_steps): 
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    cpu_rng_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        gpu_rng_state = torch.cuda.get_rng_state()
                    # optionally write model checkpoints
                    raw_model.save_pretrained(os.path.join(log_dir, f"ArabicGPT_{step:05d}"))
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict(),
                        'cpu_rng_state': cpu_rng_state
                    }
                    if torch.cuda.is_available():
                        checkpoint['gpu_rng_state'] = gpu_rng_state
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)


    # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()
