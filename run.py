from bert import Bert, Generator
from position_encoding import PositionalEncoding
from util import create_words_batch, evaluate_acc
from reader import Vocab
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
log_every_iter = 10
validate_every_iter = 2000
model_save_path = 'real_model.checkpoint'


def create_batch(batch_size, n_batches):
    for _ in range(n_batches):
        chars = torch.from_numpy(np.random.randint(2, 28, size=(batch_size, 10))).long()
        batch = Batch(chars, mask_token=mask_token, pad_token=pad_token)
        yield batch


def run_epoch(data_iter, model, loss_compute, train_iter):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):

        n_tokens = (batch.src != pad_token).sum()
        out = model.forward(batch.src, batch.src_mask)
        loss = loss_compute(out, batch.trg, batch.src_mask.squeeze(-2), n_tokens)
        total_loss += loss
        total_tokens += n_tokens
        tokens += n_tokens
        train_iter += 1

        if train_iter % log_every_iter == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / int(n_tokens), float(tokens) / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def run_simple():
    # alphabets + PAD + MASK_TOKEN
    V = 26 + 1 + 1
    d_model = 256
    h = 8

    self_attn = MultiHeadedAttention(h=h, d_model=d_model, d_k=d_model//h, d_v=d_model//h, dropout=0.3)
    feed_forward = FullyConnectedFeedForward(d_model=d_model, d_ff=1024)
    embedding = Embeddings(d_model=d_model, vocab=V)

    encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=d_model, dropout=0.1)
    generator = Generator(d_model=d_model, vocab_size=V)
    model = Bert(encoder=encoder, embedding=embedding, generator=generator, n_layers=4)
    model_opt = NoamOpt(d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(15):
        print("=" * 30)
        model.train()
        loss_compute = SimpleLossCompute(model.generator, criterion, opt=model_opt)
        train_loss = run_epoch(create_batch(30, 20), model, loss_compute)

        model.eval()
        loss_compute_eval = SimpleLossCompute(model.generator, criterion, opt=None)
        val_loss = run_epoch(create_batch(30, 5), model, loss_compute_eval)
        print("=" * 30)
        print("Epoch: %d Training Overall Loss: %f Validation Overall Loss: %f" % (epoch, float(train_loss), float(val_loss)))

    torch.save(model, "toy.model")


def read_train_data(dummy, small):
    if dummy:
        with open("words_alpha_dummy_train.txt") as f:
            words = f.read().split('\n')
    else:
        with open("words_alpha_train.txt") as f:
            words = f.read().split('\n')
    if small:
        return words[:10]
    else:
        return words[:-1]


def read_dev_data(dummy, small):
    if dummy:
        with open("words_alpha_dummy_dev.txt") as f:
            words = f.read().split('\n')
    else:
        with open("words_alpha_dev.txt") as f:
            words = f.read().split('\n')
    if small:
        return words[:10]
    else:
        return words[:-1]


def run():
    # alphabets + PAD + MASK_TOKEN
    dummy = False
    small_size = False
    use_checkpoint = True
    vocab = Vocab()
    V = len(vocab.char2id)
    d_model = 64
    d_ff = 256
    h = 4
    n_encoders = 4

    self_attn = MultiHeadedAttention(h=h, d_model=d_model, d_k=d_model//h, d_v=d_model//h, dropout=0.1)
    feed_forward = FullyConnectedFeedForward(d_model=d_model, d_ff=d_ff)
    position = PositionalEncoding(d_model, dropout=0.1)
    embedding = nn.Sequential(Embeddings(d_model=d_model, vocab=V), position)

    encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=d_model, dropout=0.1)
    generator = Generator(d_model=d_model, vocab_size=V)
    model = Bert(encoder=encoder, embedding=embedding, generator=generator, n_layers=n_encoders)

    opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(d_model, 1, 200, opt)
    criterion = nn.CrossEntropyLoss(reduction='none')
    vocab = Vocab()

    train_data = read_train_data(dummy, small_size)
    dev_data = read_dev_data(dummy, small_size)

    batch_size = 30
    train_iter = report_loss = cum_loss = valid_num = 0
    report_samples = cum_samples = 0
    hist_valid_scores = []

    if use_checkpoint:
        checkpoint = torch.load(model_save_path)
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['_step']
        rate = checkpoint['_rate']
        current_train_iter = checkpoint['train_iter']
        model_opt._step = step
        model_opt._rate = rate
        print(f'reading checkpoint from epoch {current_epoch}, iter {current_train_iter}')
    else:
        current_epoch = 0
        current_train_iter = 0

    for epoch in range(current_epoch, 100):
        print("=" * 30)
        model.train()
        loss_compute = SimpleLossCompute(model.generator, criterion, opt=model_opt)

        start = time.time()
        train_data_iter = create_words_batch(train_data, vocab, mini_batch=batch_size, shuffle=False)
        for i, batch in enumerate(train_data_iter):
            if use_checkpoint and train_iter <= current_train_iter:
                train_iter += 1
                continue
            if dummy:
                n_tokens = (batch.src != pad_token).sum(dim=1)
            else:
                n_tokens = (batch.src == mask_token).sum(dim=1)
            out = model.forward(batch.src, batch.src_mask)
            if dummy:
                batch_loss = loss_compute(out, batch.tgt, batch.src == pad_token, n_tokens)
            else:
                batch_loss = loss_compute(out, batch.tgt, batch.randomly_mask, n_tokens)

            batch_loss_val = batch_loss.item()
            report_loss += batch_loss_val
            cum_loss += batch_loss_val
            report_samples += batch_size
            cum_samples += batch_size

            train_iter += 1

            if train_iter % log_every_iter == 0:
                elapsed = time.time() - start
                print(f'epoch {epoch}, iter {train_iter}, avg. loss {report_loss / report_samples:.2f} time elapsed {elapsed:.2f}sec')
                start = time.time()
                report_loss = report_samples = 0

            if train_iter % validate_every_iter == 0:
                print(f'epoch {epoch}, iter {train_iter}, cum. loss {cum_loss / cum_samples:.2f} examples {cum_samples}')
                cum_samples = cum_loss = 0.

                print('begin evaluation...')
                valid_num += 1
                acc = evaluate_acc(model, vocab, dev_data)
                print(f'validation: iter {train_iter}, dev. acc {acc:.4f}')

                valid_metric = acc

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    print('save currently the best model to [%s]' % model_save_path)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': model_opt.optimizer.state_dict(),
                                'loss': cum_loss,
                                '_rate': model_opt._rate,
                                '_step': model_opt._step,
                                'train_iter': train_iter
                                }, model_save_path)

    torch.save(model, 'model/real.model')


run()
