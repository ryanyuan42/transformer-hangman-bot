import numpy as np
from numba.typed import List
from numba import jit, prange


def remove_one_length_word(words):
    return [w for w in words if len(np.unique(list(w))) > 1]


with open("words_alpha.txt") as f:
    words = f.read().split('\n')

mask_char = '#'

np.random.shuffle(words)


n_words = len(words)
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1


train_words = words[:int(n_words * train_ratio)]
dev_words = words[int(n_words * train_ratio): int(n_words * (train_ratio + dev_ratio))]
test_words = words[int(n_words * (train_ratio + dev_ratio)):]

train_words = remove_one_length_word(train_words)
dev_words = remove_one_length_word(dev_words)
test_words = remove_one_length_word(test_words)


with open('words_alpha_train.txt', 'w') as f:
    f.write("\n".join(train_words))

with open('words_alpha_dev.txt', 'w') as f:
    f.write('\n'.join(dev_words))

with open('words_alpha_test.txt', 'w') as f:
    f.write('\n'.join(test_words))


@jit(nopython=True, nogil=True)
def create_mask_words(words):
    result = []

    for i in prange(len(words)):
        word_lst = list(words[i])
        word_len = len(word_lst)
        n_mask = max(int(word_len * 0.4), 1)
        indices = np.random.choice(np.arange(word_len), size=n_mask)
        for j in prange(len(indices)):
            word_lst[indices[j]] = mask_char
        masked_word = ''.join(word_lst)
        result.append(masked_word)
    return result


# @jit(nopython=True, nogil=True)
def create_mask_words_unique(words):
    result = []

    for i in range(len(words)):
        word_ = list(words[i])
        word_lst = np.unique(word_)
        word_len = len(word_lst)
        n_mask = max(int(word_len * 0.5), 1)
        indices = np.random.choice(np.arange(word_len), size=n_mask)
        letters = word_lst[indices]
        for l in letters:
            for j, w in enumerate(word_):
                if w == l:
                    word_[j] = '#'
        masked_word = ''.join(word_)
        result.append(masked_word)
    return result


def create_typed_list(words):
    typed_list = List()
    for word in words:
        typed_list.append(word)
    return typed_list


def write_train_dev_test():
    train_mask_words = create_mask_words_unique(create_typed_list(train_words))
    dev_mask_words = create_mask_words_unique(create_typed_list(dev_words))
    test_mask_words = create_mask_words_unique(create_typed_list(test_words))

    with open('words_alpha_train_unique_big.txt', 'a') as f:
        for masked_word, word in zip(train_mask_words, train_words):
            f.write(','.join([masked_word, word])+'\n')
    with open('words_alpha_dev_unique_big.txt', 'a') as f:
        for masked_word, word in zip(dev_mask_words, dev_words):
            f.write(','.join([masked_word, word])+'\n')
    with open('words_alpha_test_unique_big.txt', 'a') as f:
        for masked_word, word in zip(test_mask_words, test_words):
            f.write(','.join([masked_word, word])+'\n')


write_train_dev_test()
write_train_dev_test()
write_train_dev_test()
write_train_dev_test()
write_train_dev_test()
