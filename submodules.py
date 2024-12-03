from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Model import modelArguments


def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Embedding):
        nn.init.normal_(m.weight, std=0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def tokenize(text, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=True)
    ids1 = [1] + ids + [2] 
    ids1 = torch.tensor(ids1, dtype=torch.int16)
    
    # Pad or truncate the sequence to tholtae max_seq_length
    if ids1.size(0) < modelArguments.max_seq_len:
        padding_length = modelArguments.max_seq_len - ids1.size(0)
        ids1 = torch.cat([ids1, torch.zeros(padding_length, dtype=torch.int16)])
        
    elif ids1.size(0) >= modelArguments.max_seq_len:
        ids1 = ids1[:modelArguments.max_seq_len]
        ids1[-1] = 2
    return ids1



class TokenizedIterableDataset(IterableDataset):
    def __init__(self, dataset, args, tokenizer,state_file='dataset_state.pkl'):
        self.dataset = dataset
        self.state_file = state_file
        self.buffer = ""
        self.current_index = 0
        self.args = args
        self.tokenizer = tokenizer

    def tokenize(self, text, tokenizer):
        ids = tokenizer.encode(text, add_special_tokens=True)
        ids1 = [1] + ids + [2] 
        ids1 = torch.tensor(ids1, dtype=torch.int16)
        
        # Pad or truncate the sequence to tholtae max_seq_length
        if ids1.size(0) < modelArguments.max_seq_len:
            padding_length = modelArguments.max_seq_len - ids1.size(0)
            ids1 = torch.cat([ids1, torch.zeros(padding_length, dtype=torch.int16)])
            
        elif ids1.size(0) >= modelArguments.max_seq_len:
            ids1 = ids1[:modelArguments.max_seq_len]
            ids1[-1] = 2
        return ids1

    def __iter__(self):
        index = 0
        buffer = None
        for example in self.dataset:
            # print("len of exp :",self.dataset)
            if index < self.current_index:
                index += 1
                continue
            # print(example)
            instruction = example['USER']
            output = example['ASSISTANT']

            out = instruction + "\n" + output
            out = self.tokenize(out, self.tokenizer)
            yield out
            self.current_index = index
            index += 1


        #     text = example['text']
        #     lines = text.split('.')
        #     # print("# lines :",len(lines))
        #     for line in lines:
        #         # if (buffer != None):
        #         #     yield buffer
        #         self.buffer += line 
        #         if len(self.buffer)  >= modelArguments.max_seq_len*5:
        #             tokenized_input = tokenize(self.buffer)

                    
        #             yield tokenized_input
        #             self.buffer = line 


        #     self.current_index = index
        #     # self.save_state()
        #     index += 1

        # # Tokenize the remaining buffer if it's not empty
        # if len(self.buffer) > 0:
        #     tokenized_input = tokenize(self.buffer)
        #     if tokenized_input is not None:
        #         yield tokenized_input




    def save_state(self):
        state = {'buffer': self.buffer, 'current_index': self.current_index}
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
                self.buffer = state.get('buffer', "")
                self.current_index = state.get('current_index', 0)


def collate_fn(batch):
        inputs = torch.stack([item for item in batch])
        return inputs


def id_to_token(x, tokenizer):
    s = ''
    for i in x[0]:
        # j = torch.argmax(i)
        s += tokenizer.decode(i)

    return s

def id_to_token_M(x, tokenizer):
    s = ''
    for i in x[0]:
        j = torch.argmax(i)
        s += tokenizer.decode(j)

    return s


class TransformerLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        step = max(step, 1)  # To avoid division by zero
        return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)


def plot_loss_with_trend(loss_array, window_size=25):
    """
    Plots the loss function with an additional line to show the trend.

    Args:
    loss_array (list or np.array): Array of loss values.
    window_size (int): The window size for calculating the moving average trend line.
    """
    # Convert loss_array to a numpy array if it's not already
    loss_array = np.array(loss_array)

    # Calculate the moving average for the trend line
    trend_line = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')
    trend = (trend_line[0]-trend_line[-1])/trend_line.size
    trend2 = (trend_line[-1]-trend_line[-window_size-1])/window_size if len(trend_line) > window_size else 0
    

    # Plot the loss function
    print("overall :",trend)
    print("last window :",trend2)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_array, label='Loss', color='blue', alpha=0.6)
    plt.plot(range(window_size - 1, len(loss_array)), trend_line, label='Trend', color='red', linestyle='--', linewidth=2)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.yscale('log')
    plt.title('Loss Function with Trend Line')
    plt.legend()
    plt.grid(True)
    plt.savefig('current_loss.jpg')
    # plt.show()