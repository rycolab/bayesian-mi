import torch.nn as nn


class TransparentDataParallel(nn.DataParallel):

    def set_best(self, *args, **kwargs):
        return self.module.set_best(*args, **kwargs)

    def recover_best(self, *args, **kwargs):
        return self.module.recover_best(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)

    def train_batch(self, *args, **kwargs):
        return self.module.train_batch(*args, **kwargs)

    def eval_batch(self, *args, **kwargs):
        return self.module.eval_batch(*args, **kwargs)

    def get_norm(self, *args, **kwargs):
        return self.module.get_norm(*args, **kwargs)

    def get_rank(self, *args, **kwargs):
        return self.module.get_rank(*args, **kwargs)
