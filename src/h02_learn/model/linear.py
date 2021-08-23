import math
import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel


class Linear(BaseModel):
    # pylint: disable=too-many-instance-attributes,arguments-differ

    name = 'linear'

    def __init__(self, task, embedding_size=768, n_classes=3, alpha=0.0,
                 dropout=0.1, representation=None, n_words=None):
        super().__init__()

        # Save things to the model here
        self.alpha = alpha
        self.dropout_p = dropout
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.representation = representation
        self.n_words = n_words
        self.task = task

        if self.representation in ['onehot', 'random']:
            self.build_embeddings(n_words, embedding_size)

        self.linear = nn.Linear(embedding_size, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

    def build_embeddings(self, n_words, embedding_size):
        if self.task == 'dep_label':
            self.embedding_size = int(embedding_size / 2) * 2
            self.embedding = nn.Embedding(n_words, int(embedding_size / 2))
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        if self.representation == 'random':
            self.embedding.weight.requires_grad = False

    def forward(self, x, eps=1e-5):
        if self.representation in ['onehot', 'random']:
            x = self.get_embeddings(x)
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

        x_emb = self.dropout(x)
        logits = self.linear(x_emb)
        return logits

    def get_embeddings(self, x):
        x_emb = self.embedding(x)
        if len(x.shape) > 1:
            x_emb = x_emb.reshape(x.shape[0], -1)

        return x_emb

    def train_batch(self, data, target, optimizer):
        optimizer.zero_grad()
        mlp_out = self(data)
        loss = self.get_loss(mlp_out, target)
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy.item()

    def get_loss(self, predicted, target):
        entropy = self.criterion(predicted, target) / math.log(2)
        penalty = self.get_norm()
        return entropy + self.alpha * penalty

    def get_norm(self):
        ext_matrix = torch.cat([self.linear.weight, self.linear.bias.unsqueeze(-1)], dim=1)
        penalty = torch.norm(ext_matrix, p='nuc')

        return penalty

    def get_args(self):
        return {
            'alpha': self.alpha,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'n_classes': self.n_classes,
            'representation': self.representation,
            'n_words': self.n_words,
            'task': self.task,
        }

    @staticmethod
    def print_param_names():
        return [
            'alpha', 'embedding_size', 'dropout',
            'n_classes', 'representation', 'n_words',
        ]

    def print_params(self):
        return [
            self.alpha, self.embedding_size, self.dropout_p,
            self.n_classes, self.representation, self.n_words
        ]
