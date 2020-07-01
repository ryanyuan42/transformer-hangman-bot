import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedFeedForward(nn.Module):
    """
    A fully connected neural network with Relu activation
    input: d_model
    hidden: d_ff
    output: d_model

    Implements FFN equation.
    FFN(x) = max(0, xW_1 + b)W_2 + b

    It consist of two linear layer and a Relu activation in between

    Linear_2(Relu(Linear_1(x))))
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FullyConnectedFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: shape (batch_size, max_sent_len, embedding_size/d_model)
        :return: output: shape (batch_size, max_sent_len, embedding_size/d_model)
        """
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
