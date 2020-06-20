import torch


class NoamOpt:
    """
    Optim wrapper that implements rate.

    vary the learning rate during training

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    that is linearly increasing learning rate linearly during warm up and decreasing it
    thereafter proportionally to the inverse root of the step number
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup

        self._step = 0
        self._rate = 0

    def step(self):
        """
        Update parameters and rate and step the optimizer
        """

        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):

        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

