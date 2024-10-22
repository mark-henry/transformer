import math

import torch.nn as nn
from torch._C._te import Tensor

# %%
import numpy as np
import torch


def positional_encoding(seq_len, d_model):
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return torch.FloatTensor(pos_encoding)


# %%
class Attention(nn.Module):
    def __init__(self, seq_len, model_dimension, key_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.model_dimension = model_dimension
        self.key_dimension = key_dimension
        self.Q = nn.Linear(model_dimension, key_dimension)
        self.K = nn.Linear(model_dimension, key_dimension)
        self.V = nn.Linear(model_dimension, model_dimension)

        mask = torch.ones(self.seq_len, self.seq_len)
        # Make it lower triangular and invert
        self.mask: Tensor = torch.tril(mask) == 0

    def forward(self, input):
        scale = math.sqrt(self.key_dimension)
        attention_pattern = torch.matmul(self.Q(input), self.K(input).T) / scale
        masked_attn = attention_pattern.masked_fill(self.mask, float('-inf'))
        attention = nn.Softmax(dim=1)(masked_attn)
        return torch.matmul(attention, self.V(input))


class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, seq_len, num_attention_heads, hidden_layer_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.num_attention_heads = num_attention_heads
        self.attention_heads = [
            Attention(seq_len, embedding_size,
                      embedding_size // num_attention_heads)
            for _ in range(num_attention_heads)
        ]
        self.layer_norm = nn.LayerNorm([seq_len, embedding_size])
        self.hidden_layer_size = hidden_layer_size or 4 * embedding_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, embedding_size)
        )

    def forward(self, input):
        attention_values = sum(head(input) for head in self.attention_heads)
        attn_add_and_norm = self.layer_norm(input + attention_values)
        ff_value = self.feed_forward(attn_add_and_norm)
        return self.layer_norm(attn_add_and_norm + ff_value)


class Transformer(nn.Module):
    def __init__(self, embedding_size, vocab_size, pad_token_id, seq_len=64, num_attention_heads=8, num_layers=6, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        decoder_layers = nn.Sequential(
            *[DecoderLayer(embedding_size, seq_len, num_attention_heads, *args, **kwargs) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            decoder_layers,
            nn.Linear(embedding_size, vocab_size),
            nn.Softmax(dim=1)
        )
        self.pad_token_id = pad_token_id

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        pe = positional_encoding(self.seq_len, self.embedding_size)
        encoded = embedded + pe
        return self.decoder(encoded)

