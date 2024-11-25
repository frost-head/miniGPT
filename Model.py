import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math 
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.cuda.amp import autocast

@dataclass
class modelArguments:
    d_model = 768
    n_layers = 6
    n_heads = 16
    vocab_size = 32768
    device = "cuda"
    max_seq_len = 512
    max_batch_size  = 4
    dropout = 0.1


class shared_emb(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model

        # Create a single shared embedding
        self.shared_weight = nn.Parameter(torch.randn(vocab_size, d_model))

        # Initialize the weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.shared_weight)

    def embedding(self, x):
        # Scale embeddings by sqrt(d_model)
        return nn.functional.embedding(x, self.shared_weight) * math.sqrt(self.d_model)

    def linear(self, x):
        # Use the transpose of the shared weight for the pre-softmax linear transformation
        return nn.functional.linear(x, self.shared_weight)



class FlashAttention(nn.Module):
    def __init__(self, args : modelArguments):
        super(FlashAttention, self).__init__()
        self.embed_dim = args.d_model
        self.num_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        self.args = args
        self.q_linear = nn.Linear(args.d_model, args.d_model, bias=False)
        self.k_linear = nn.Linear(args.d_model, args.d_model, bias=False)
        self.v_linear = nn.Linear(args.d_model, args.d_model, bias=False)
        self.out_linear = nn.Linear(args.d_model, args.d_model, bias=False)

        assert args.d_model % args.d_model == 0, "Embedding dimension must be divisible by the number of heads."

    def forward(self, query, key, value, mask=False):
        batch_size, seq_len, _ = query.size()

        # Linear projections 
        q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q = q.contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        # k = k.contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        # v = v.contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):    
            attn_output = F.scaled_dot_product_attention(q,k,v,dropout_p=modelArguments.dropout, is_causal=mask)

        # Reshape output
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final linear layer
        output = self.out_linear(attn_output)

        return output

class FeedForward(nn.Module):
    def __init__(self, args: modelArguments) -> None:
        super().__init__()

        self.d_model = args.d_model
        self.hidden_dim = 4*self.d_model

        self.fc1 = nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, self.d_model, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args : modelArguments) -> None:
        super().__init__()
        self.args = args
        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.attention = FlashAttention(args)
        self.feedforward = FeedForward(args)

    def forward(self, x : torch.tensor):
        embd = self.attention(x,x,x)
        embd = self.norm1(embd)
        cat =  F.dropout(embd + x, self.args.dropout)
        embd = self.feedforward(cat)
        embd = self.norm2(embd)
        return  F.dropout(embd + cat, self.args.dropout)



def get_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    pos_encoding = torch.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding


class model(nn.Module):
    def __init__(self, args : modelArguments) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab size should be set."

        self.args = args
        self.shared_emb = shared_emb(args.vocab_size, args.d_model)
        # self.enc_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.enc_layers = nn.ModuleList()

        for _ in range(args.n_layers):
            self.enc_layers.append(EncoderBlock(args))


        self.norm = nn.LayerNorm(args.d_model)
        # self.norm1 = nn.LayerNorm(args.d_model)
        # self.linear = nn.Linear(args.d_model, args.vocab_size, bias = False)
        self.enc = get_positional_encoding(args.max_seq_len, args.d_model).to(args.device)
        # self.softmax = nn.Softmax(-1)


    def forward(self, in_token : torch.tensor):

        # print(in_token.shape)
        inp_embd = self.shared_emb.embedding(in_token)
        # inp_embd = F.dropout(inp_embd, self.args.dropout)
        # print(self.enc.shape)
        # print(inp_embd.shape)
        inp_embd = inp_embd + self.enc
        # print(freq.shape)
        # prev_layer = inp_embd
        for layer in self.enc_layers:
            inp_embd = layer(inp_embd)
            inp_embd = F.dropout(inp_embd, self.args.dropout)
            # inp_embd = inp_embd + prev_layer
            # prev_layer = inp_embd
            inp_embd = self.norm(inp_embd)


        # out_embd = self.shared_emb.embedding(out_token)
        # print(out_embd[0])
        # print(self.enc[0])
        # # out_embd = F.dropout(out_embd, self.args.dropout)
        # out_embd = out_embd + self.enc
        # # print(freq.shape)
        # prev_layer = out_embd

        # for layer in self.dec_layers:
        #     print(layer)
        #     out_embd = layer(out_embd, inp_embd)

        #     out_embd = out_embd + prev_layer
        #     out_embd = F.dropout(out_embd, self.args.dropout)
        #     prev_layer = out_embd
        #     out_embd = self.norm(out_embd)


        output = self.shared_emb.linear(inp_embd)

        return output