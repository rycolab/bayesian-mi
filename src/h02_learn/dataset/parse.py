import numpy as np
# import pandas as pd
import torch
# from torch.utils.data import Dataset

# from h01_data.process import get_data_file_base as get_file_names
from .base import BaseDataset
# from util import util


class ParseDataset(BaseDataset):
    name = 'parse'

    def _process_index(self, classes, words):
        x_raw, y_raw = self.load_data(self.iterate_index)

        self.load_index(x_raw, words=words)
        self.load_classes(y_raw, classes=classes)

    def _process(self, classes):
        x_raw, y_raw = self.load_data(self.iterate_embeddings)

        self.load_embeddings(x_raw)
        self.load_classes(y_raw, classes=classes)

    def _load_data(self, iterator):
        x_raw, y_raw = [], []
        for sentence_ud, sentence_tokens in iterator():
            # Add root
            if isinstance(sentence_tokens[0], np.ndarray):
                x_sentence = [np.zeros(sentence_tokens[0].shape)]
            else:
                x_sentence = ['<ROOT>']

            y_sentence = [-1]

            for i, token in enumerate(sentence_ud):
                head = token['head']

                if head is None:
                    continue

                x_sentence += [sentence_tokens[i]]
                y_sentence += [head]

            x_raw += [np.array(x_sentence)]
            y_raw += [np.array(y_sentence)]

        return x_raw, y_raw

    def load_index(self, x_raw, words=None):
        if words is None:
            words = []
        all_words = {token for sentence in x_raw for token in sentence}
        new_words = sorted(list(all_words - set(words)))
        if new_words:
            words = np.concatenate([words, new_words])

        words_dict = {word: i for i, word in enumerate(words)}
        x = [np.array([[words_dict[token]] for token in tokens]) for tokens in x_raw]

        self.x = [torch.from_numpy(sentence) for sentence in x]
        self.words = words

        self.n_words = len(words)

    def load_embeddings(self, x_raw):
        self.assert_size(x_raw)
        self.x = [torch.from_numpy(x) for x in x_raw]

    def assert_size(self, x):
        assert x[0].shape[-1] == self.embedding_size

    def load_classes(self, y_raw, classes=None):
        self.y = [torch.from_numpy(y) for y in y_raw]

        self.classes = None
        self.n_classes = 0
