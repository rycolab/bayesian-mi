import numpy as np
import torch

from .pos_tag import PosTagDataset


class DepLabelDataset(PosTagDataset):
    name = 'dep_label'

    def load_index(self, x_raw, words=None):
        if words is None:
            words = []

        new_words = sorted(list(set(np.unique(x_raw)) - set(words)))
        if new_words:
            words = np.concatenate([words, new_words])

        words_dict = {word: i for i, word in enumerate(words)}
        x = np.array([[words_dict[token] for token in tokens] for tokens in x_raw])

        self.x = torch.from_numpy(x)
        self.words = words

        self.n_words = len(words)

    def _load_data(self, iterator):
        x_raw, y_raw = [], []
        for sentence_ud, sentence_tokens in iterator():
            for i, token in enumerate(sentence_ud):
                head = token['head']
                rel = token['rel']

                if rel in {"_", "root"}:
                    continue

                x_raw_tail = sentence_tokens[i]
                x_raw_head = sentence_tokens[head - 1]

                x_raw += [[x_raw_tail, x_raw_head]]
                y_raw += [rel]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        if len(x_raw.shape) == 3:
            x_raw = x_raw.reshape(x_raw.shape[0], -1)  # pylint: disable=E1136  # pylint/issues/3139

        return x_raw, y_raw
