import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# from h01_data.process import get_data_file_base as get_file_names
from .base import BaseDataset
# from util import util


class PosTagDataset(BaseDataset):
    name = 'pos_tag'

    def load_index(self, x_raw, words=None):
        self.x, self.words = self.factorize(x_raw, words)
        self.n_words = len(self.words)

    @staticmethod
    def _load_data(iterator):
        x_raw, y_raw = [], []
        for sentence_ud, sentence_tokens in iterator():
            for i, token in enumerate(sentence_ud):
                pos_tag = token['pos']

                if pos_tag in {"_", "X"}:
                    continue

                x_raw += [sentence_tokens[i]]
                y_raw += [pos_tag]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw
