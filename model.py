import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, List

# Text Embedding - Softmax
class TextEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # create embedding feature
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):

        embeded_x = self.embedding(x) * math.sqrt(self.d_model)
        # embeded_x = embeded_x.view(embeded_x.size(1), embeded_x.size(0), -1)
        return embeded_x

# Positional Encoding (fixed version)
class PositionalEncodeing(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

        # create empty encoding matrix with shape [max_len, d_model]
        PE = torch.zeros(max_len, d_model)

        # position vector: tensor [[0], [1],..., [max_len-1]]
        p = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        # take 2*i index of embedding vector: tensor [0, 2, 4, ...]
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32)

        # 10000**(2*i/d_model)
        # 1/x^n = x^(-n)
        # x^y = e^(y * ln(x))
        div_term = torch.exp(-two_i / d_model * math.log(10000.0))

        # assign each embedding vec of position encode
        # sin for even index of embedding vectors (d_model) of each position of PE
        PE[:, 0::2] = torch.sin(p * div_term)
        # cos for odd index of embedding vectors (d_model) of each position of PE
        PE[:, 1::2] = torch.cos(p * div_term)
        
        # Add batch dimension: shape from [max_len, d_model]-> [1, max_len, d_model]
        PE = PE.unsqueeze(0).requires_grad_(False)

        # make pos encode fixed, not change during training
        self.register_buffer('PE', PE)
    
    def forward(self, x: torch.Tensor):

        # x has [batch_size, max_len, dim_embed], pe has [batch_size, max_len, d_model]
        pe = self.PE
        
        # combine with input x
        
        x += pe

        # dropout
        x = self.dropout(x)

        return x

# Add & Norm
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):

        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Position-wise Feed-Forward Networks
class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # W1, b1
        self.linear_1 = nn.Linear(d_model, d_ff)
        # W2, b2
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # max(0, xW1 + b1)W2 + b2 

        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

# Transform query, key, value to queries, keys, values for heads of multi-head attention
class TransformInput(nn.Module):
    def __init__(self, d_model: int, heads: int, head_dim: int, bias: bool = True):
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim
        self.linear = nn.Linear(d_model, heads * head_dim, bias = bias)

    def forward(self, x: torch.Tensor):

        # input of x will be has shape [ batch_size, seq_len, d_model]
        head_shape = x.shape[:-1] #  [ batch_size, seq_len]

        
        # cal QW or KW or VW
        x = self.linear(x)

        # reshape x into [batch_size, seq_len, heads, head_dim]
        x = x.view(*head_shape, self.heads, self.head_dim)

        return x

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()

        self.heads = heads
        # dim of each head
        self.head_dim = d_model // heads

        # cal QW, KW, VW or queries, keys, values for each head from Q, K, V
        # all will have shape = [batch_size, seq_len, heads, head_dim]
        self.queries = TransformInput(d_model, heads, self.head_dim, bias = bias)
        self.keys = TransformInput(d_model, heads, self.head_dim, bias = bias)
        self.values = TransformInput(d_model, heads, self.head_dim, bias = bias)

        # components of Scaled Dot-Product Attention
        # Scale
        self.scale = 1 / math.sqrt(self.head_dim)
        # Softmax
        self.softmax = nn.Softmax(dim=1)

        # components of Multi-Head Attention
        # Linear layer
        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    # check shape Mask == shape Q @ K^T
    # mask is the same to all heads
    def Mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):

        # mask = mask.squeeze(1)
        
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[-1] == key_shape[1]
        assert mask.shape[1] == 1 or mask.shape[-1] == query_shape[1]
        
        return mask
    

    def forward(self, *, # * mean you must put the follow arguments with its keyword
                query: torch.Tensor, #  query, key and value all have shape [seq_len, batch_size, d_model]
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        seq_len, batch_size, _= query.shape
        

        # all will have shape = [batch_size, seq_len, heads, head_dim]
        queries = self.queries(query)
        keys = self.keys(key)
        values = self.values(value)

        # Q @ K_T
        # batch_size, seq_len, heads, d_k = queries.shape
        # scores is [batch_size, heads, seq_len, seq_len], so
        # scores = torch.zeros(batch_size, heads, seq_len, seq_len)
        # for i in range(seq_len):
        #     for j in range(seq_len):
        #         for b in range(batch_size):
        #             for h in range(heads):
        #                 scores[i, j, b, h] = (queries[i, b, h] * keys[j, b, h]).sum()
        # using einsum, life easier :))
        scores = torch.einsum('bihd,bjhd->bhij', queries, keys)

        # scale
        scores *= self.scale

        # apply mask for the output target encode
        if mask is not None:

            mask = self.Mask(mask, query.shape, key.shape)

            scores = scores.masked_fill(mask==0, -1e9)
        
        attn = self.softmax(scores)

        attn = self.dropout(attn)

        # mulmat with values
        # x will have same shape with "values"
        x = torch.einsum("bhij,bjhd->bihd", attn, values)

        # concat
        x = x.reshape(seq_len, batch_size, -1)

        # Liner
        x = self.output(x)

        return x 

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

        self.multi_head_attn = MultiHeadAttention(heads= heads, d_model=d_model, dropout=dropout, bias=bias)
        self.norm_attn = LayerNorm()
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm_ff = LayerNorm()

    def forward(self, *,
                x: torch.Tensor,
                x_mask: torch.Tensor):
        
        x_norm = self.norm_attn(x)
        attn = self.multi_head_attn(query=x_norm, key=x_norm, value=x_norm, mask=x_mask)
        residual_attn = x + self.dropout(attn)

        attn_norm = self.norm_ff(residual_attn)
        ff = self.feed_forward(attn_norm)
        residual_ff =  residual_attn + self.dropout(ff)

        return residual_ff

# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

        self.masked_multi_head_attn = MultiHeadAttention(heads= heads, d_model=d_model, dropout=dropout, bias=bias)
        self.norm_masked_attn = LayerNorm()
        self.multi_head_attn = MultiHeadAttention(heads= heads, d_model=d_model, dropout=dropout, bias=bias)
        self.norm_attn = LayerNorm()
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm_ff = LayerNorm()

    def forward(self, *,
                encode_x: torch.Tensor,
                x_mask: torch.Tensor,
                target_x: torch.Tensor,
                target_mask: torch.Tensor):
        
        target_x_norm = self.norm_masked_attn(target_x)
        masked_attn = self.masked_multi_head_attn(query=target_x_norm, key=target_x_norm, value=target_x_norm, mask=target_mask)
        residual_masked_attn = target_x + self.dropout(masked_attn)
        
        masked_x_norm = self.norm_attn(residual_masked_attn)
        encode_x_norm = self.norm_attn(encode_x)
        attn = self.multi_head_attn(query=masked_x_norm, key=encode_x_norm, value=encode_x_norm, mask=x_mask)
        residual_attn = masked_x_norm + self.dropout(attn)

        attn_norm = self.norm_ff(residual_attn)
        ff = self.feed_forward(attn_norm)
        residual_ff =  residual_attn + self.dropout(ff)

        return residual_ff

class Encoder(nn.Module):
    def __init__(self, num_layers: int, heads:int, d_model:int, d_ff:int, dropout:float, bias:bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([EncoderLayer(heads=heads, d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias) for i in range(num_layers)])
        self.layer_norm = LayerNorm()

    def forward(self, x:torch.Tensor, x_mask:torch.Tensor):

        for i in range(self.num_layers):
            
            x = self.layers[i](x=x, x_mask=x_mask)

        return self.layer_norm(x)

class Decoder(nn.Module):
    def __init__(self, num_layers: int, heads:int, d_model:int, d_ff:int, dropout:float, bias:bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DecoderLayer(heads=heads, d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias) for i in range(num_layers)])
        self.layer_norm = LayerNorm()

    def forward(self,
                encode_x: torch.Tensor,
                x_mask: torch.Tensor,
                target_x: torch.Tensor,
                target_mask: torch.Tensor):
        
        for i in range(self.num_layers):
            x = self.layers[i](encode_x=encode_x, x_mask=x_mask, target_x=target_x, target_mask=target_mask)

        return self.layer_norm(x)  

# Output softmax for decoder
class OutputGenerator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = True):
        super().__init__()

        self.linear = nn.Linear(d_model, vocab_size, bias = bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, *,
                x: torch.Tensor):
        
        x = self.linear(x)
        softmax_output = self.softmax(x)

        return softmax_output

# Transformer
class Transformer(nn.Module):
    def __init__(self, 
                 max_len: int,
                 input_vocab_size: int,
                 output_vocab_size: int,
                 num_layers: int,  
                 heads: int, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1, 
                 bias: bool = True):
        
        super().__init__()

        self.heads = heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

        self.input_embedding = TextEmbedding(d_model=d_model, vocab_size=input_vocab_size)
        self.output_embedding = TextEmbedding(d_model=d_model, vocab_size=output_vocab_size)
        self.positional_encoding = PositionalEncodeing(d_model=d_model, dropout=dropout, max_len=max_len)
        self.encoder = Encoder(num_layers=num_layers, heads=heads, d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)
        self.decoder = Decoder(num_layers=num_layers, heads=heads, d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)
        self.generator = OutputGenerator(d_model=d_model, vocab_size=output_vocab_size, bias=bias)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                x_target: torch.Tensor,
                target_mask: torch.Tensor):
        
        input_embeded = self.input_embedding(x) #[batch_size, max_len, dim_embed]
        output_embeded = self.output_embedding(x_target) #[batch_size, max_len, dim_embed]

        encoder_input = self.positional_encoding(input_embeded) #[batch_size, max_len, dim_embed]
        decoder_input = self.positional_encoding(output_embeded) #[batch_size, max_len, dim_embed]

        encoder_output = self.encoder(x=encoder_input, x_mask=x_mask) #[batch_size, max_len, dim_embed]
        
        #[batch_size, max_len, dim_embed]
        decoder_output = self.decoder(encode_x=encoder_output, 
                                      x_mask=x_mask, 
                                      target_x=decoder_input, 
                                      target_mask=target_mask)

        softmax_output = self.generator(x=decoder_output)  #[batch_size, max_len, vocab_size]

        return softmax_output
    
