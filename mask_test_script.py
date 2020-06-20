from bert import Bert, Generator
import time
import torch.nn as nn
import torch
from encoder import Encoder
from attention import MultiHeadedAttention
from feed_forward import FullyConnectedFeedForward
from model_embeddings import Embeddings
import numpy as np
from loss_function import SimpleLossCompute
from optimimizer import NoamOpt
from batch import Batch

pad_token = 0
mask_token = 1


def create_batch(batch_size, n_batches):
    for _ in range(n_batches):
        chars = torch.from_numpy(np.random.randint(2, 28, size=(batch_size, 10))).long()
        batch = Batch(chars, None, pad_token)
        yield batch


V = 26 + 1 + 1
d_model = 256
h = 8

self_attn = MultiHeadedAttention(h=h, d_model=d_model, d_k=d_model//h, d_v=d_model//h, dropout=0.)
feed_forward = FullyConnectedFeedForward(d_model=d_model, d_ff=1024)
embedding = Embeddings(d_model=d_model, vocab=V)

encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=d_model, dropout=0.)
generator = Generator(d_model=d_model, vocab_size=V)
model = Bert(encoder=encoder, embedding=embedding, generator=generator, n_layers=4)

data_iter = create_batch(30, 5)
for i, batch in enumerate(data_iter):
    x = embedding(batch.src)
    y = self_attn(x, x, x, batch.src_mask)

    masked_src = batch.src.masked_fill(batch.src_mask.squeeze(-2) == 0, mask_token)
    x2 = embedding(masked_src)
    y2 = self_attn(x2, x2, x2, batch.src_mask)
    print(y)
