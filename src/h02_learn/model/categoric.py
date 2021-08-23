import math
import torch
import torch.nn as nn

from .base import BaseModel


class Categoric(BaseModel):
    # pylint: disable=too-many-instance-attributes,arguments-differ

    name = 'categoric'

    def __init__(self, task, n_classes=100):
        # pylint: disable=too-many-arguments
        super().__init__()

        self.task = task
        self.n_classes = n_classes
        self.alpha = 2

        self.probs = nn.Parameter(torch.Tensor(self.n_classes))
        self.log_probs = nn.Parameter(torch.Tensor(self.n_classes))
        self.count = nn.Parameter(
            torch.LongTensor(self.n_classes).zero_(),
            requires_grad=False)

        self.criterion = nn.NLLLoss(ignore_index=self.ignore_index)

    def fit(self, trainloader):
        with torch.no_grad():
            for _, y in trainloader:
                self.fit_batch(_, y)

    def fit_batch(self, _, y):
        for char in y.unique():
            if char == self.ignore_index:
                continue
            self.count[char] += (y == char).sum()

        self.probs[:] = \
            (self.count.float() + self.alpha) / (self.count.sum() + self.alpha * self.n_classes)
        self.log_probs[:] = torch.log(self.probs)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.task == 'parse':
            max_len = x.shape[1]
            y_hat = self.log_probs[:max_len] \
                .reshape(1, 1, -1) \
                .repeat(batch_size, max_len, 1)
        else:
            y_hat = self.log_probs \
                .reshape(1, -1) \
                .repeat(batch_size, 1)

        return y_hat

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().detach().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy

    def get_args(self):
        return {
            'n_classes': self.n_classes,
            'task': self.task,
        }

    @staticmethod
    def print_param_names():
        return [
            'n_classes', 'task'
        ]

    def print_params(self):
        return [
            self.n_classes, self.task
        ]
