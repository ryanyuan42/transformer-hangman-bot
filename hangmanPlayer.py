import torch
import string
from reader import Vocab
import torch.nn.functional as F
from bert import Bert, Generator
from encoder import Encoder
from attention import MultiHeadedAttention
from feed_forward import FullyConnectedFeedForward
from position_encoding import PositionalEncoding
from model_embeddings import Embeddings
from hangmanUtil import get_best_first_char
import numpy as np
import torch.nn as nn


class HangmanPlayer:
    def __init__(self):
        self.guessed_letters = []
        self.vocab = Vocab()
        self.model = self.load_model('model_v1/real_model.checkpoint')

    def guess_vowel(self, question):
        return get_best_first_char(len(question), self.guessed_letters)

    def guess(self, question):
        if question.count("#") == len(question):
            pred = self.guess_vowel(question)
            self.guessed_letters.append(pred)
            return pred

        guessed = [self.vocab.char2id[l] for l in self.guessed_letters]
        p = self.get_most_prob(question)
        p[guessed] = -np.inf

        pred = self.vocab.id2char[np.argmax(p)]
        self.guessed_letters.append(pred)
        return pred

    def guess2(self, question):
        for c in string.ascii_lowercase:
            if c not in self.guessed_letters:
                self.guessed_letters.append(c)
                return c

    def new_game(self):
        self.guessed_letters = []

    def load_model(self, path):
        device = torch.device('cpu')
        V = len(self.vocab.char2id)
        d_model = 64
        d_ff = 256
        h = 4
        n_encoders = 4

        self_attn = MultiHeadedAttention(h=h, d_model=d_model, d_k=d_model // h, d_v=d_model // h, dropout=0.1)
        feed_forward = FullyConnectedFeedForward(d_model=d_model, d_ff=d_ff)
        position = PositionalEncoding(d_model, dropout=0.1)
        embedding = nn.Sequential(Embeddings(d_model=d_model, vocab=V), position)

        encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=d_model, dropout=0.1)
        generator = Generator(d_model=d_model, vocab_size=V)
        model = Bert(encoder=encoder, embedding=embedding, generator=generator, n_layers=n_encoders)
        model = model.to(device)
        model_save_path = path
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def get_most_prob(self, masked_word):
        with torch.no_grad():
            src = torch.tensor([[self.vocab.char2id[c] for c in masked_word]])

            src_mask = ((src != self.vocab.char2id['#']) & (src != self.vocab.char2id['_'])).unsqueeze(-2)
            out = self.model.forward(src, src_mask)
            p = F.softmax(self.model.generator(out), dim=2).squeeze(0)
            return torch.max(p[(src == self.vocab.char2id['#']).squeeze(0)], dim=0).values.detach().numpy()
