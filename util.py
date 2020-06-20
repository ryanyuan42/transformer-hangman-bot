import torch.nn as nn
from reader import Vocab
import torch
from batch import Batch
import numpy as np
import copy


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


def pad_words(words, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    :param words: (list[list[int]]): list of words, where each sentence
                                    is represented as a list of words
    :param pad_token: (int): padding token
    :returns words_padded: (list[list[int]]): list of sentences where sentences shorter
        Output shape: (batch_size, max_word_length)
    """

    word_lens = [len(word) for word in words]
    max_len = max(word_lens)
    words_padded = [sent + [pad_token] * (max_len - word_lens[i]) for i, sent in enumerate(words)]

    return words_padded


def evaluate_acc(model, vocab, dev_data):
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    total_correct_guess = 0
    total_n = 0
    with torch.no_grad():
        for batch in create_words_batch(dev_data, vocab, mini_batch=30, shuffle=False):

            output = model(batch.src, batch.src_mask)
            p = model.generator(output)
            correct_guess = ((torch.argmax(p, dim=2) == batch.tgt) * (batch.randomly_mask == 0)).sum()
            all_masked = (batch.randomly_mask == 0).sum()

            total_correct_guess += correct_guess.item()
            total_n += all_masked.item()

        acc = total_correct_guess / total_n

    if was_training:
        model.train()

    return acc


def create_words_batch(lines, vocab, mini_batch: int, shuffle=True):
    if shuffle:
        np.random.shuffle(lines)

    src_buffer = []
    tgt_buffer = []
    for line in lines:
        src_word, tgt_word = line.split(',')
        if len(line) > 1:
            src_buffer.append([vocab.char2id[c] for c in src_word])
            tgt_buffer.append([vocab.char2id[c] for c in tgt_word])

            if len(src_buffer) == mini_batch:
                src = torch.tensor(pad_words(src_buffer, vocab.char2id['_'])).long()
                tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id['_'])).long()

                batch = Batch(src, tgt, mask_token=vocab.char2id['#'], pad_token=vocab.char2id['_'])
                yield batch
                src_buffer = []
                tgt_buffer = []

    if len(src_buffer) != 0:
        src = torch.tensor(pad_words(src_buffer, vocab.char2id['_'])).long()
        tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id['_'])).long()

        batch = Batch(src, tgt, mask_token=vocab.char2id['#'], pad_token=vocab.char2id['_'])
        yield batch



def read_train_data():
    with open("words_alpha_train.txt") as f:
        words = f.read().split('\n')
    return words


if __name__ == "__main__":
    vocab = Vocab()
    train_data = read_train_data()
    batches = create_words_batch(train_data, vocab, 30, shuffle=False)

    all_train_data = [b.src.shape[0] for b in batches]
    print(len(all_train_data))
    print(sum(all_train_data))
