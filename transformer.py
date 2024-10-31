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
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)) == 0
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, input, padding_mask=None):
        batch_size = input.shape[0]
        scale = math.sqrt(self.key_dimension)
        attention_pattern = torch.bmm(self.Q(input), self.K(input).transpose(1, 2)) / scale

        mask = self.causal_mask.expand(batch_size, -1, -1)

        if padding_mask is not None:
            # Mask attention TO padded tokens
            mask = mask | padding_mask.unsqueeze(1)
            # Mask attention FROM padded tokens
            mask = mask | padding_mask.unsqueeze(2)

        # Apply combined mask
        attention_pattern = attention_pattern.masked_fill(mask, float('-inf'))

        # Check for any rows that are completely masked
        all_neg = (attention_pattern == float('-inf')).all(dim=-1)
        if all_neg.any():
            # For rows that are all masked, create a uniform distribution
            attention_pattern = attention_pattern.clone()
            attention_pattern[all_neg] = 0.0

        attention = nn.Softmax(dim=-1)(attention_pattern)
        return torch.bmm(attention, self.V(input))


class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, seq_len, num_attention_heads, ff_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.num_attention_heads = num_attention_heads
        key_dim = embedding_size // num_attention_heads  # AIAYN recommends d_k = d_v = d_model/h
        self.attention_heads = nn.ModuleList([
            Attention(seq_len, embedding_size, key_dim)
            for _ in range(num_attention_heads)
        ])
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        ff_size = ff_size or 4 * embedding_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embedding_size)
        )

    def forward(self, input, padding_mask=None):
        attention_values = torch.stack([head(input, padding_mask) for head in self.attention_heads]).mean(0)
        attn_add_and_norm = self.layer_norm1(input + attention_values)
        return self.layer_norm2(attn_add_and_norm + self.feed_forward(attn_add_and_norm))


class Transformer(nn.Module):
    def __init__(self, embedding_size, vocab_size, pad_token_id,
                 embedding=None, seq_len=64, num_attention_heads=8, num_layers=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.pad_token_id = pad_token_id
        self.register_buffer('pe', self._create_positional_encoding(seq_len, embedding_size))
        self.embedding = embedding
        if not embedding:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            # Claude recommended this initialization
            nn.init.normal_(self.embedding.weight, mean=0, std=embedding_size ** -0.5)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_size, seq_len, num_attention_heads, *args, **kwargs)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(embedding_size, vocab_size)

    @staticmethod
    def _create_positional_encoding(seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, token_ids):
        padding_mask = (token_ids == self.pad_token_id)

        embedded = self.embedding(token_ids)
        encoded = embedded + self.pe.unsqueeze(0)

        hidden_state = encoded
        for layer in self.decoder_layers:
            hidden_state = layer(hidden_state, padding_mask)

        return self.output_projection(hidden_state)