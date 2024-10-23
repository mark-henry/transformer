import math
import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, seq_len, model_dimension, key_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.model_dimension = model_dimension
        self.key_dimension = key_dimension
        self.Q = nn.Linear(model_dimension, key_dimension)
        self.K = nn.Linear(model_dimension, key_dimension)
        self.V = nn.Linear(model_dimension, model_dimension)
        mask = torch.ones(seq_len, seq_len)
        self.register_buffer('mask', torch.tril(mask) == 0)

    def forward(self, input):
        batch_size = input.shape[0]
        scale = math.sqrt(self.key_dimension)
        attention_pattern = torch.bmm(self.Q(input), self.K(input).transpose(1, 2)) / scale
        masked_attn = attention_pattern.masked_fill(self.mask.expand(batch_size, -1, -1), float('-inf'))
        attention = nn.Softmax(dim=-1)(masked_attn)
        return torch.bmm(attention, self.V(input))

class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, seq_len, num_attention_heads, ff_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.num_attention_heads = num_attention_heads
        key_dim = embedding_size // num_attention_heads  # AIAYN recommends d_k = d_v = d_model/h = 64
        self.attention_heads = nn.ModuleList([
            Attention(seq_len, embedding_size, key_dim)
            for _ in range(num_attention_heads)
        ])
        self.layer_norm = nn.LayerNorm([seq_len, embedding_size])
        ff_size = ff_size or 4 * embedding_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embedding_size)
        )

    def forward(self, input):
        attention_values = torch.stack([head(input) for head in self.attention_heads]).sum(0)
        attn_add_and_norm = self.layer_norm(input + attention_values)
        return self.layer_norm(attn_add_and_norm + self.feed_forward(attn_add_and_norm))

class Transformer(nn.Module):
    def __init__(self, embedding_size, vocab_size, pad_token_id, seq_len=64, num_attention_heads=8, num_layers=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.register_buffer('pe', self._create_positional_encoding(seq_len, embedding_size))
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        decoder_layers = nn.Sequential(
            *[DecoderLayer(embedding_size, seq_len, num_attention_heads, *args, **kwargs) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            decoder_layers,
            nn.Linear(embedding_size, vocab_size),
            nn.Softmax(dim=-1)
        )
        self.pad_token_id = pad_token_id

    @staticmethod
    def _create_positional_encoding(seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        encoded = embedded + self.pe.unsqueeze(0)
        return self.decoder(encoded)