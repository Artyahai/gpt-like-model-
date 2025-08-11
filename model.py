import torch
import torch.nn as nn 
import math
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class InputEmbeddings(nn.Module):
    def __init__(self,vocab_size:int, d_model:int)->None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
  def __init__(self, seq_len:int, d_model:int, dropout:float = 0.1)->None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  def forward(self, x:torch.tensor):
        x = x+self.pe[:, :x.size(1), :]
        return self.dropout(x)

def make_causal_masks(seq_len, device):
    return torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

def combine_masks(pad_mask, causal_mask):
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        return pad_mask * causal_mask
    return causal_mask


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float = 0.1)->None:
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.dropout = nn.Dropout(dropout)        
        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    @staticmethod
    def attention(query, key, value,mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    def forward(self,q,k,v,mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, attention_scores = MaskedMultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class LayerNormalization(nn.Module):
    def __init__(self, d_model:int,eps:float=1e-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self, x:torch.Tensor):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
class ResidualConnection(nn.Module):
    def __init__(self, d_model:int, dropout:float = 0.1)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float = 0.1)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1)->None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
    def forward(self, x, pad_mask = None):
        seq_len = x.size(1)
        casual_mask = make_causal_masks(seq_len, x.device)
        mask = combine_masks(pad_mask, casual_mask)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x



class GptModel(nn.Module):
    def __init__(self,seq_len:int,vocab_size:int, d_model:int,num_layers, d_ff:int,h:int, dropout:float=0.1)->None:
        super().__init__()
        self.embeddings = InputEmbeddings(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(seq_len, d_model, dropout)
        self.layers = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, x, pad_mask=None):
        x = self.embeddings(x)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, pad_mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits           


    


    

    
