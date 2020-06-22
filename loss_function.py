import torch
import numpy as np


class SimpleLossCompute:

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        if self.opt:
            self.opt.optimizer.zero_grad()

    def __call__(self, x, y, mask, norm):
        batch_size = x.shape[0]
        x = self.generator(x)
        # loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                       y.contiguous().view(-1)) / norm
        loss_across = self.criterion(x.transpose(1, 2), y)
        loss = ((loss_across * (mask == 0)).sum(dim=1) / norm).sum() / batch_size

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * batch_size


class KLLossCompute:

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        if self.opt:
            self.opt.optimizer.zero_grad()

    def __call__(self, x, y, mask):
        batch_size = x.shape[0]
        x = self.generator(x)
        # x = x.masked_fill(mask == 1, -np.inf)
        # loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                       y.contiguous().view(-1)) / norm
        batch_loss = self.criterion(x, y)
        loss = batch_loss / batch_size

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return batch_loss.data


class KLLossComputeMasked:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        if self.opt:
            self.opt.optimizer.zero_grad()

    def __call__(self, x, y, mask):
        batch_size = x.shape[0]
        x = self.generator(x, mask)
        # x = x.masked_fill(mask == 1, -np.inf)
        # loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                       y.contiguous().view(-1)) / norm
        batch_loss = self.criterion(x, y)
        loss = batch_loss / batch_size

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return batch_loss.data