import math

import torch.nn as nn
from torch._C._te import Tensor
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')




vocab_size = tokenizer.vocab_size
embedding_dim = config.hidden_size
embedding = nn.Embedding(vocab_size, embedding_dim)

example_text = ["Hello, how are you today?"]
encoded_inputs = tokenizer(example_text, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded_inputs['input_ids']
embedded = embedding(input_ids)

# %%
import numpy as np
import torch

def positional_encoding(max_seq_len, d_model):
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = np.zeros((max_seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return torch.FloatTensor(pos_encoding)

# %%
class Attention(nn.Module):
    def __init__(self, seq_len, model_dimension, key_dimension, values_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.model_dimension = model_dimension
        self.key_dimension = key_dimension
        self.values_dimension = values_dimension
        self.Q = nn.Linear(model_dimension, key_dimension)
        self.K = nn.Linear(model_dimension, key_dimension)
        self.V = nn.Linear(model_dimension, model_dimension)

        mask = torch.ones(self.seq_len, self.seq_len)
        # Make it lower triangular and invert
        self.mask: Tensor = torch.tril(mask) == 0

    def forward(self, input):
        scale = math.sqrt(self.key_dimension)
        attention_pattern = torch.matmul(self.K(input), self.Q(input)) / scale
        masked_attn = attention_pattern.masked_fill(self.mask, float('-inf'))
        attention = nn.Softmax(dim=1)(attention_pattern)
        return self.V(attention)


class Transformer(nn.Module):
    def __init__(self, embedding_size: int, vocab_size, max_seq_len=64, num_attention_heads=8, *args, **kwargs):
        """
        :model_dimension: the dimensionality of the embeddings
        """
        super().__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention_heads = [Attention(max_seq_len, embedding_size, ?, ?) for _ in range(num_attention_heads)]

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        pe = positional_encoding(self.max_seq_len, self.d_model)
        pe_out = embedded + pe
        attention_values = [head(pe_out) for head in self.attention_heads]


