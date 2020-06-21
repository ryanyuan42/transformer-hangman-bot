import torch.nn as nn
from numba import jit
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


def evaluate_acc(model, vocab, dev_data, device):
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    total_correct_guess = 0
    total_n = 0
    with torch.no_grad():
        for batch in create_words_batch(dev_data, vocab, mini_batch=30, shuffle=False, device=device):

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


def evaluate_acc_v2(model, vocab, dev_data, device):
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    total_correct_guess = 0
    total_n = 0
    with torch.no_grad():
        for batch in create_words_batch_v2(dev_data, vocab, mini_batch=30, shuffle=False, device=device):

            output = model(batch.src, batch.src_mask)
            p = model.generator(output)
            most_prob_letter = torch.argmax(p, dim=1)
            most_prob_mask = torch.zeros(batch.tgt.shape, device=device)
            most_prob_mask = most_prob_mask.scatter_(1, most_prob_letter.view(-1, 1), 1)
            batch_guess = (batch.tgt * most_prob_mask).sum(dim=1)

            total_correct_guess += (batch_guess != 0).sum().item()
            total_n += batch_guess.shape[0]

        acc = total_correct_guess / total_n

    if was_training:
        model.train()

    return acc


def convert_target_to_dist(target, vocab, mask, device):
    dist_mask = torch.zeros(target.shape[0], len(vocab.char2id), device=device)
    dist_mask = dist_mask.scatter_(1, target * mask, 1)

    target_numpy = (target * mask).numpy()
    extra_col = np.ones((target.shape[0], 1), dtype=target_numpy.dtype) * (vocab.char2id['z'] + 1)
    target_numpy = np.hstack((target_numpy, extra_col))
    target_np_dist = np.apply_along_axis(np.bincount, 1, target_numpy)[:, :-1]
    target_dist = torch.from_numpy(target_np_dist)
    target_dist[:, 0] = 0

    return target_dist * dist_mask


def create_words_batch(lines, vocab, mini_batch: int, device, shuffle=True):
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
                src = torch.tensor(pad_words(src_buffer, vocab.char2id['_']), device=device).long()
                tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id['_']), device=device).long()

                batch = Batch(src, tgt, mask_token=vocab.char2id['#'], pad_token=vocab.char2id['_'])
                yield batch
                src_buffer = []
                tgt_buffer = []

    if len(src_buffer) != 0:
        src = torch.tensor(pad_words(src_buffer, vocab.char2id['_']), device=device).long()
        tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id['_']), device=device).long()

        batch = Batch(src, tgt, mask_token=vocab.char2id['#'], pad_token=vocab.char2id['_'])
        yield batch


def create_words_batch_v2(lines, vocab, mini_batch: int, device, shuffle=True):
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
                src = torch.tensor(pad_words(src_buffer, vocab.char2id['_']), device=device)
                tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id['_']), device=device)
                tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id['#'], device=device)
                tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None])

                batch = Batch(src, tgt_dist, mask_token=vocab.char2id['#'], pad_token=vocab.char2id['_'])
                yield batch
                src_buffer = []
                tgt_buffer = []

    if len(src_buffer) != 0:
        src = torch.tensor(pad_words(src_buffer, vocab.char2id['_']), device=device)
        tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id['_']), device=device)
        tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id['#'], device=device)
        tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None])

        batch = Batch(src, tgt_dist, mask_token=vocab.char2id['#'], pad_token=vocab.char2id['_'])
        yield batch


def read_train_data():
    with open("words_alpha_train.txt") as f:
        words = f.read().split('\n')
    return words


if __name__ == "__main__":
    device = torch.device("cpu")
    vocab = Vocab()
    train_data = read_train_data()
    batches = create_words_batch(train_data, vocab, 30, shuffle=False, device=device)

    all_train_data = [b.src.shape[0] for b in batches]
    print(len(all_train_data))
    print(sum(all_train_data))
