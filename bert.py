import torch.nn as nn
import torch.nn.functional as F
import torch
from layer_norm import LayerNorm
from util import clone


class Bert(nn.Module):
    def __init__(self, encoder: nn.Module, generator, embedding, n_layers: int):
        """

        :param encoder: encoder/transformer layer that takes advantage of self-attention
        :param n_layers: int, number of encoder/transformer layers
        """
        super(Bert, self).__init__()
        self.encoder = encoder
        self.layers = clone(encoder, n_layers)
        self.embed = embedding
        self.layer_norm = LayerNorm(encoder.size)
        self.generator = generator

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        """
        :param x: shape (batch_size, max_word_length)
        :param src_mask
        :return:
        """
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)

    @property
    def device(self):
        return self.generator.linear.weight.device


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(in_features=d_model,
                                out_features=vocab_size,
                                )

    def forward(self, x):
        return self.linear(x)
    

class Generator2(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator2, self).__init__()
        self.linear = nn.Linear(in_features=d_model,
                                out_features=vocab_size)

    def forward(self, x):
        result, _ = torch.max(self.linear(x), dim=1)
        return F.log_softmax(result, dim=1)

