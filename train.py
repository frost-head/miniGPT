import torch
import torch.nn as nn
from Model import model, modelArguments
import time
import re
import math
import triton
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast
import matplotlib.pyplot as plt
import numpy as np
import os
from submodules import *

CUDA_LAUCH_BLOCKING = 1
TORCH_USE_CUDA_DSA = True


model_weights = './checkpoints/model_weightsBPE0.pth' # weights for model
optim_weights = './checkpoints/optimizer0.optimizer0.pth' # optimizer Weights
history_weights = './checkpoints/history0.pth' # history of training. 
tokenizer_path = './large_tokenizer.json'



print("Setting matmul precision to medium ......")
torch.set_float32_matmul_precision('medium') # Increases traning speed with minimal performance impact
 
print("Loading Model .....")
fmodel = model(modelArguments) # Object creation


for name, para in fmodel.named_parameters(): #layers of model
    print(name)

print("Model have ",count_parameters(fmodel), "parameters")




if os.path.exists(model_weights) == True:
    # applying weights
    print("Weights Found .... \nApplying Weights .....")

    state_dict = torch.load(model_weights)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    fmodel.load_state_dict(new_state_dict, strict=False)

    print("Application Successfull .....")
else:
    print("No Weights Found ..... \nInitializing Model Weights .....")
    fmodel.apply(initialize_weights)
    print("done")


print("Model to {moelArguments.device}")
fmodel = fmodel.to(modelArguments.device)





print("Loading dataset ......")

# Dataset 1
# dataset = load_dataset("TIGER-Lab/MATH-plus", split="train", streaming=False)

# Dataset 2
# dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)

# Dataset 3
dataset = load_dataset("rombodawg/LosslessMegaCodeTrainingV2", streaming=False, split="train")
print("Dataset Loaded ....")


assert(os.path.exists(tokenizer_path))
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
print("Tokenizer initialized .....")


train_dataset = TokenizedIterableDataset(dataset,modelArguments, tokenizer) #creating a iterable Dataset

train_dataloader = DataLoader(train_dataset, batch_size=modelArguments.max_batch_size, collate_fn=collate_fn) # Create a DataLoader




if os.path.exists(history_weights):
    # applying weights
    print("History Found .... \nApplying Weights .....")

    history = torch.load(history_weights)
    print("Application Successfull .....")

else:
    print("No History Found ..... \nInitializing Model Weights .....")
    history = []
    print("done")


loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(fmodel.parameters(), betas=(0.9, 0.99), lr = 3e-4)


if os.path.exists(optim_weights):
    # applying weights
    print("optimizer Found .... \nApplying Weights .....")

    optimizer.load_state_dict(torch.load('checkpoints/optimizer0.pth'))

    print("Application Successfull .....")


optimizer.zero_grad()
torch.cuda.empty_cache()
# schedular = TransformerLRScheduler(optimizer, 512, 500)


rloss = 0

open = False
for epoch in range(4000):
    global_step = 0
    sloss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=modelArguments.max_batch_size, collate_fn=collate_fn)
    for batch_idx, a in enumerate(train_dataloader):

        a = a.to(torch.long).to('cuda')

        max_no_tokens = 2**7       # accumulation_steps2 = 1
        accumulation_steps = max_no_tokens//(modelArguments.max_batch_size)



        t0 = time.time()
        out = fmodel(a)
        loss = loss_fn(out[:, :modelArguments.max_seq_len-1, :].contiguous().view(-1, out.size(-1)), a[:, 1:modelArguments.max_seq_len].contiguous().view(-1)) 
        loss = loss / accumulation_steps



        rloss += loss.item()
        loss.backward()


        if (batch_idx+1) % ((max_no_tokens / modelArguments.max_batch_size)*32) == 0 or math.isnan(loss):
            
            torch.save(fmodel.state_dict(), 'checkpoints/model_weightsBPE0.pth')
            torch.save(optimizer.state_dict(), 'checkpoints/optimizer0.pth')
            torch.save(history, 'checkpoints/history0.pth')
            plot_loss_with_trend(history[0000:], window_size=1000)        
            os.system('clear')
            print("-"*6,f" Loss: {rloss}", "-"*6,"\n")
            print("out: ", id_to_token(a, tokenizer),"\n")

            print("out: ", id_to_token_M(out, tokenizer),"\n")


            print('\n')

            
        
        if (batch_idx+1) % accumulation_steps == 0:
            
            torch.nn.utils.clip_grad_norm_(fmodel.parameters(), max_norm=1)
            optimizer.step()
            # schedular.step()
            # scaler.update()
            global_step += 1
            optimizer.zero_grad()

            history.append(rloss)
            
            torch.cuda.synchronize()
            t1 = time.time()

            dt = (t1-t0)*1000 
        
            print(f"-----Steps : {global_step}, BatchIDX : {batch_idx}, Loss: {rloss:8f}, #tokens : {max_no_tokens} ,timeT : {dt}ms ----- ")
    
            rloss = 0
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} completed, {batch_idx}")   


