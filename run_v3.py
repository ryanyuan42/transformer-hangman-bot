from bert import Bert, Generator, Generator2, Generator3
import os
from position_encoding import PositionalEncoding
from util import create_words_batch_v2, evaluate_acc_v3
from reader import Vocab
import time
import torch.nn as nn
import torch
from encoder import Encoder
from attention import MultiHeadedAttention
from feed_forward import FullyConnectedFeedForward
from model_embeddings import Embeddings
from loss_function import SimpleLossCompute, KLLossComputeMasked
from optimimizer import NoamOpt
pad_token = 0
mask_token = 1
log_every_iter = 100
validate_every_iter = 10000


def read_train_data(dummy, small):
    if dummy:
        with open("words_alpha_dummy_train.txt") as f:
            words = f.read().split('\n')
    else:
        with open("words_alpha_train_unique_big.txt") as f:
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
        with open("words_alpha_dev_unique.txt") as f:
            words = f.read().split('\n')
    if small:
        return words[:10]
    else:
        return words[:-1]


def run_v3():
    directory = 'model_v3/'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    model_save_path = 'real_model.checkpoint'
    model_save_path = os.path.join(directory, model_save_path)

    dummy = False
    small_size = False
    use_checkpoint = False
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocab = Vocab()
    V = len(vocab.char2id)
    d_model = 256
    d_ff = 1024
    h = 4
    n_encoders = 4

    self_attn = MultiHeadedAttention(h=h, d_model=d_model, d_k=d_model // h, d_v=d_model // h, dropout=0.1)
    feed_forward = FullyConnectedFeedForward(d_model=d_model, d_ff=d_ff)
    position = PositionalEncoding(d_model, dropout=0.1)
    embedding = nn.Sequential(Embeddings(d_model=d_model, vocab=V), position)

    encoder = Encoder(self_attn=self_attn, feed_forward=feed_forward, size=d_model, dropout=0.1)
    generator = Generator3(d_model=d_model, vocab_size=V)
    model = Bert(encoder=encoder, embedding=embedding, generator=generator, n_layers=n_encoders)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(d_model, 2, 4000, opt)
    criterion = nn.KLDivLoss(reduction='sum')
    if use_cuda:
        criterion.cuda(device=device)
    vocab = Vocab()

    train_data = read_train_data(dummy, small_size)
    dev_data = read_dev_data(dummy, small_size)

    batch_size = 64
    train_iter = report_loss = cum_loss = valid_num = 0
    report_samples = cum_samples = 0
    hist_valid_scores = []

    if use_checkpoint:
        checkpoint = torch.load(model_save_path)
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
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
        loss_compute = KLLossComputeMasked(model.generator, criterion, opt=model_opt)

        start = time.time()
        train_data_iter = create_words_batch_v2(train_data, vocab, mini_batch=batch_size, shuffle=False,
                                                device=model.device)
        for i, batch in enumerate(train_data_iter):
            if use_checkpoint and train_iter <= current_train_iter:
                train_iter += 1
                continue
            out = model.forward(batch.src, batch.src_mask)
            generator_mask = torch.zeros(batch.src.shape[0], V, device=model.device)
            generator_mask = generator_mask.scatter_(1, batch.src, 1)

            if dummy:
                batch_loss = loss_compute(out, batch.tgt, generator_mask)
            else:
                batch_loss = loss_compute(out, batch.tgt, generator_mask)

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
                acc = evaluate_acc_v3(model, vocab, dev_data, device=model.device)
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
                                'train_iter': train_iter,
                                'hist_valid_scores': hist_valid_scores,
                                }, model_save_path)

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': cum_loss,
                    '_rate': model_opt._rate,
                    '_step': model_opt._step,
                    'train_iter': train_iter,
                    'hist_valid_scores': hist_valid_scores,
                    }, os.path.join(directory, f'real_model_{epoch}.checkpoint'))


run_v3()
