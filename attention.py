import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


def attention(query, key, value, mask=None, dropout=None):
    """
    query: shape (*, n_queries, d_k) n_queries is the maximum sentence length / max_sent_length - 1 if key from decoder
    key: (*, K, d_k) , K is the maximum sentence length / max_sent_length - 1 if key from decoder
    value: (*, K, d_v)

    scores: (n_quires, K)
    output: (n_queries, d_v)
    """

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -np.inf)
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)

    return torch.matmul(p, value), p


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.query_linear = nn.Linear(in_features=d_k * h,
                                      out_features=d_model,
                                      bias=False)
        self.key_linear = nn.Linear(in_features=d_k * h,
                                    out_features=d_model,
                                    bias=False)
        self.value_linear = nn.Linear(in_features=d_v * h,
                                      out_features=d_model,
                                      bias=False)

        self.attn = None  # not used for computation, only for visualization
        self.dropout = nn.Dropout(p=dropout)

        self.output_linear = nn.Linear(in_features=d_model,
                                       out_features=h * d_v)

    def forward(self, query, key, value, mask=None):
        """
        d_k * h = d_model

        query: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size,
        key: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size
        value: shape (batch_size, max_sent_length, embedding_size), d_model is the embedding size,

        output: shape (batch_size, max_sent_length, embedding_size)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        d_k = self.d_model // self.h

        n_batches = query.size(0)
        max_sent_length = query.size(1)

        query = self.query_linear(query).view(n_batches, max_sent_length, self.h, d_k).transpose(1, 2)
        key = self.key_linear(key).view(n_batches, key.size(1), self.h, d_k).transpose(1, 2)
        value = self.value_linear(value).view(n_batches, value.size(1), self.h, d_k).transpose(1, 2)

        # scores shape: (batch_size, h, max_sent_length, d_k)
        scores, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concat attention scores over multiple heads
        # (batch_size, max_sent_length, d_model)
        scores = scores.transpose(1, 2).contiguous().view(n_batches, max_sent_length, self.h * d_k)

        return self.output_linear(scores)
