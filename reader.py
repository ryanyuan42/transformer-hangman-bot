import string

class Vocab:
    def __init__(self):
        self.char2id = dict()
        self.char2id['_'] = 0
        self.char2id['#'] = 1
        self.char_list = string.ascii_lowercase
        for i, c in enumerate(self.char_list):
            self.char2id[c] = len(self.char2id)
        self.id2char = {v: k for k, v in self.char2id.items()}


if __name__ == "__main__":
    vocab = Vocab()

    batches = [batch.src.shape[0] for batch in vocab.create_words_batch(30, shuffle=True)]
    print(sum(batches))

