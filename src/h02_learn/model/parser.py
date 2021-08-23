import math
import numpy as np
import torch
import torch.nn as nn

from .linear import Linear
from .mlp import MLP


class LinearParser(Linear):
    # pylint: disable=too-many-instance-attributes,arguments-differ
    name = 'linear'

    def __init__(self, embedding_size=768, alpha=0.0,
                 dropout=0.1, representation=None, n_words=None):
        super().__init__('parse', embedding_size=embedding_size, n_classes=1, alpha=alpha,
                         dropout=dropout, representation=representation, n_words=n_words)

        self.embedding_size = embedding_size
        # self.linear = nn.Linear(embedding_size * 2, 1)
        self.biaffine = Biaffine(embedding_size, embedding_size, use_linears=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        del self.linear

    def forward(self, x, eps=1e-5):
        if self.representation in ['onehot', 'random']:
            sent_lens = (x != 0).sum(-1)
            x = self.get_embeddings(x)
        else:
            sent_lens = (x != -1).any(-1).sum(-1)
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
        x_emb = self.dropout(x)

        # sent_len = x_emb.shape[1]
        # x_l = x_emb.unsqueeze(2).repeat(1, 1, sent_len, 1)
        # x_r = x_emb.unsqueeze(1).repeat(1, sent_len, 1, 1)

        # x_in = torch.cat([x_l, x_r], dim=-1)
        # logits = self.linear(x_in).squeeze(-1)
        logits = self.biaffine(x_emb, x_emb)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            logits[i, sent_len:, :] = 1e-9
            logits[i, :, sent_len:] = 1e-9
        return logits

    def get_embeddings(self, x):
        x_emb = self.embedding(x)
        return x_emb

    def get_loss(self, predicted, target):
        entropy = self.criterion(
            predicted.reshape(-1, predicted.shape[-1]),
            target.reshape(-1)) / math.log(2)
        if self.alpha == 0:
            return entropy

        penalty = self.get_norm()
        return entropy + self.alpha * penalty

    def get_norm(self):
        # ext_matrix = torch.cat([self.biaffine.matrix], dim=1)
        penalty = torch.norm(self.biaffine.matrix, p='nuc')

        return penalty

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(
            mlp_out.reshape(-1, mlp_out.shape[-1]),
            target.reshape(-1)) / math.log(2)
        accuracy = self.get_uas(mlp_out, target)
        loss = loss.item() * data.shape[0]

        return loss, accuracy.item()

    @staticmethod
    def get_uas(predicted, target):
        matches = (predicted.argmax(dim=-1) == target)
        valid = (target != -1)
        valid[:, 0] = False
        sentence_accuracy = (matches & valid).float().sum(-1) / valid.sum(-1)
        accuracy = sentence_accuracy.sum()
        return accuracy


class MLPParser(MLP, LinearParser):
    # pylint: disable=too-many-instance-attributes,arguments-differ, super-init-not-called

    def __init__(self, embedding_size=768, hidden_size=5,
                 nlayers=1, dropout=0.1, representation=None, n_words=None):
        # pylint: disable=too-many-arguments
        MLP.__init__(self, 'parse', embedding_size=embedding_size, n_classes=1,
                     hidden_size=hidden_size, nlayers=nlayers, dropout=dropout,
                     representation=representation, n_words=n_words)

        self.embedding_size = embedding_size

        self.mlp_dep = self.mlp
        self.mlp_arc = self.build_mlp()
        self.biaffine = Biaffine(self.final_hidden_size, self.final_hidden_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        del self.mlp
        del self.out

    def forward(self, x):
        if self.representation in ['onehot', 'random']:
            sent_lens = (x != 0).sum(-1)
            x = self.get_embeddings(x)
        else:
            sent_lens = (x != -1).any(-1).sum(-1)

        x_emb = self.dropout(x)

        h_dep = self.mlp_dep(x_emb)
        h_arc = self.mlp_arc(x_emb)

        logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            logits[i, sent_len:, :] = 1e-9
            logits[i, :, sent_len:] = 1e-9
        return logits

    def get_embeddings(self, x):
        return LinearParser.get_embeddings(self, x)

    def eval_batch(self, *args):
        return LinearParser.eval_batch(self, *args)

    def train_batch(self, data, target, optimizer):
        optimizer.zero_grad()
        mlp_out = self(data)
        loss = self.criterion(mlp_out.reshape(-1, mlp_out.shape[-1]), target.reshape(-1))
        loss.backward()
        optimizer.step()

        return loss.item() / math.log(2)


class Biaffine(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, dim_left, dim_right, use_linears=True):
        super().__init__()
        self.dim_left = dim_left
        self.dim_right = dim_right
        self.use_linears = use_linears

        self.matrix = nn.Parameter(torch.Tensor(dim_left, dim_right))
        self.bias = nn.Parameter(torch.Tensor(1))

        if use_linears:
            self.linear_l = nn.Linear(dim_left, 1)
            self.linear_r = nn.Linear(dim_right, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.matrix)

    def forward(self, x_l, x_r):
        # x shape [batch, length_l, length_r]
        x = torch.matmul(x_l, self.matrix)
        x = torch.bmm(x, x_r.transpose(1, 2)) + self.bias

        # x shape [batch, length_l, 1] and [batch, 1, length_r]
        if self.use_linears:
            x += self.linear_l(x_l) + self.linear_r(x_r).transpose(1, 2)
        return x
