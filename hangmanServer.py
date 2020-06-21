import torch.nn as nn
import pandas as pd
from collections import defaultdict
from typing import List
import numpy as np
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
with open("words_alpha_train.txt") as f:
    words = f.read().split('\n')
words = [word.split(',')[1] for word in words[:-1]]

words_df = pd.Series(words).to_frame('word')
words_df['letters'] = words_df['word'].apply(lambda x: list(x))
words_df['word_len'] = words_df['letters'].apply(lambda x: len(x))


class HangmanServer:
    def __init__(self, player):
        self.player = player
        self.test_words = []

    @staticmethod
    def read_test_words():
        with open("words_alpha_test.txt") as f:
            words = f.read().split('\n')

        return words[:-1]

    @staticmethod
    def data_iter(words):
        for word in words:
            _, answer = word.split(',')
            question = '#' * len(answer)
            yield question, answer

    def run(self):
        test_words = self.read_test_words()
        np.random.shuffle(test_words)
        test_words = test_words[:1000]
        qa_pair = self.data_iter(test_words)
        success = total = 0
        success_rate = 0
        print(f"Total Game Number: {len(test_words)}")
        for question, answer in qa_pair:
            self.player.new_game()
            tries = 6
            success_rate = 0 if total == 0 else success / total
            print("=" * 20, "Game %d" % (total + 1), '=' * 20, "Success Rate: %.2f" % success_rate)
            # if (total + 1) % 100 == 0:
            #     print(total + 1)
            while '#' in question and tries > 0:
                guess = self.player.guess(question)
                print("provided question: ", " ".join(question), "your guess: %s" % guess, "left tries: %d" % tries, 'answer: %s' % answer)
                question_lst = []
                for q_l, a_l in zip(question, answer):
                    if q_l == '#':
                        if a_l == guess:
                            question_lst.append(a_l)
                        else:
                            question_lst.append(q_l)
                    else:
                        question_lst.append(q_l)
                question = "".join(question_lst)
                if guess not in answer:
                    tries -= 1

            if '#' not in question:
                success += 1
            total += 1

        print(f"{success} success out of {total} tries, rate: {success / total:.4f}")


class HangmanPlayer:
    def __init__(self):
        self.guessed_letters = []
        self.vocab = Vocab()
        self.model = self.load_model('real_model.checkpoint')

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
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def get_most_prob(self, masked_word):
        with torch.no_grad():
            src = torch.tensor([[self.vocab.char2id[c] for c in masked_word]])

            src_mask = ((src != self.vocab.char2id['#']) & (src != self.vocab.char2id['_'])).unsqueeze(-2)
            out = self.model.forward(src, src_mask)
            p = F.softmax(self.model.generator(out), dim=2).squeeze(0)
            return torch.max(p[(src == self.vocab.char2id['#']).squeeze(0)], dim=0).values.detach().numpy()


def get_best_first_char(word_len: int, guessed_letters: List[str]) -> str:
    """
    Get the letter when no letters has been guessed in the word.
    The idea is that most words have vowels in them, therefore, vowels will be a
    good starting point.
    1. Get the most frequent letter in the same length words.
    2. Get the most frequent letter in the same length words condition on the words
    don't have the vowels that have been already guessed
    :param word_len: int, word length
    :param guessed_letters: list, the letters that have been guessed
    :return: str, the letter to guess
    """
    global words_df
    words_n_df = words_df[words_df['word_len'] == word_len]
    cond = pd.Series([True] * len(words_n_df), index=words_n_df.index)
    for letter in guessed_letters:
        cond &= words_n_df['word'].apply(lambda x: letter not in x)

    letter_freq = count_vowels_freq(words_n_df[cond]['word'])
    if letter_freq:
        rank = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
        for letter, _ in rank:
            if letter not in guessed_letters:
                return letter
    else:
        letter_freq = count_letter_freq(words_n_df[cond]['word'])
        if not letter_freq:
            letter_freq = count_letter_freq(words_n_df['word'])
        rank = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
        return rank[0][0]


def count_vowels_freq(words: List[str]):
    letter_freq = defaultdict(int)
    for char in 'aeiou':
        for word in words:
            if char in word:
                letter_freq[char] += 1
    return letter_freq


def count_letter_freq(words: List[str], normalize=False):
    letter_freq = defaultdict(int)
    for char in string.ascii_lowercase:
        for word in words:
            if char in word:
                letter_freq[char] += 1
    if normalize:
        for char in letter_freq:
            letter_freq[char] /= len(words)
    return letter_freq


if __name__ == "__main__":
    player = HangmanPlayer()
    server = HangmanServer(player)

    server.run()
