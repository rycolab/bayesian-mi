from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from h01_data.process import get_data_file_base as get_file_names
from util import util


class BaseDataset(Dataset, ABC):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data_path, language, representation, embedding_size,
                 mode, classes=None, words=None, max_instances=None):
        self.data_path = data_path
        self.language = language
        self.mode = mode
        self.representation = representation
        self.embedding_size = embedding_size
        self.max_instances = max_instances

        self.input_name_base = get_file_names(data_path, language)
        self.process(classes, words)

        assert len(self.x) == len(self.y)
        self.n_instances = len(self.x)

    def process(self, classes, words):
        if self.representation not in ['onehot', 'random']:
            self._process(classes)
            self.words = words
            self.n_words = None
        else:
            self._process_index(classes, words)

    def _process_index(self, classes, words):
        x_raw, y_raw = self.load_data(self.iterate_index)

        self.load_index(x_raw, words=words)
        self.load_classes(y_raw, classes=classes)

    def _process(self, classes):
        x_raw, y_raw = self.load_data(self.iterate_embeddings)

        self.load_embeddings(x_raw)
        self.load_classes(y_raw, classes=classes)

    def load_data(self, iterator):
        x_raw, y_raw = self._load_data(iterator)
        if self.max_instances is not None:
            x_raw = x_raw[:self.max_instances]
            y_raw = y_raw[:self.max_instances]
        return x_raw, y_raw

    @abstractmethod
    def _load_data(self, iterator):
        pass

    def iterate_index(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))

        for (sentence_ud, words) in data_ud:
            yield sentence_ud, np.array(words)

    def iterate_embeddings(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))
        data_embeddings = util.read_data(self.input_name_base % (self.mode, self.representation))

        for (sentence_ud, _), sentence_emb in zip(data_ud, data_embeddings):
            yield sentence_ud, sentence_emb

    def load_embeddings(self, x_raw):
        self.assert_size(x_raw)
        self.x = torch.from_numpy(x_raw)

    def assert_size(self, x):
        assert len(x[0]) == self.embedding_size

    @abstractmethod
    def load_index(self, x_raw, words=None):
        pass

    def load_classes(self, y_raw, classes=None):
        self.y, self.classes = self.factorize(y_raw, classes)
        self.n_classes = self.classes.shape[0]

    def factorize(self, data_raw, classes=None):
        if self.mode != 'train':
            assert classes is not None

        if classes is None:
            data, classes = pd.factorize(data_raw, sort=True)
        else:
            new_classes = set(data_raw) - set(classes)
            if new_classes:
                classes = np.concatenate([classes, list(new_classes)])

            classes_dict = {pos_class: i for i, pos_class in enumerate(classes)}
            data = np.array([classes_dict[token] for token in data_raw])

        return torch.from_numpy(data), classes

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
