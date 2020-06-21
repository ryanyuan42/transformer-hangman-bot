import pandas as pd
from typing import List
from collections import defaultdict
import string
with open("words_alpha_train.txt") as f:
    words = f.read().split('\n')
words = [word.split(',')[1] for word in words[:-1]]

words_df = pd.Series(words).to_frame('word')
words_df['letters'] = words_df['word'].apply(lambda x: list(x))
words_df['word_len'] = words_df['letters'].apply(lambda x: len(x))


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
