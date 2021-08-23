import torch
from torch.utils.data import DataLoader

from util import constants
from util import util
from .pos_tag import PosTagDataset
from .dep_label import DepLabelDataset
from .parse import ParseDataset


def batch_generator(task):
    def generate_batch(batch):
        x = torch.cat([item[0].unsqueeze(0) for item in batch], dim=0)
        y = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0)

        x, y = x.to(device=constants.device), y.to(device=constants.device)
        return (x, y)

    if task in ['pos_tag', 'dep_label']:
        return generate_batch

    def pad_batch(batch):
        batch_size = len(batch)
        max_length = max([len(sentence[0]) for sentence in batch])
        shape = batch[0][0].shape[-1]

        x = torch.ones(batch_size, max_length, shape) * -1
        y = torch.ones(batch_size, max_length).long() * -1

        for i, sentence in enumerate(batch):
            sent_len = len(sentence[0])
            x[i, :sent_len] = sentence[0]
            y[i, :sent_len] = sentence[1]

        if shape == 1:
            x = x.squeeze(-1).long()
            x[x == -1] = 0

        x, y = x.to(device=constants.device), y.to(device=constants.device)
        return (x, y)

    if task in ['parse']:
        return pad_batch

    raise ValueError('Invalid task for batch generation')


def get_data_cls(task):
    if task == 'pos_tag':
        return PosTagDataset
    if task == 'dep_label':
        return DepLabelDataset
    if task == 'parse':
        return ParseDataset

    raise ValueError('Invalid task %s' % task)


def get_data_loader(dataset_cls, task, data_path, language, representations, embedding_size,
                    mode, batch_size, shuffle, classes=None, words=None, max_instances=None):
    # pylint: disable=too-many-arguments
    trainset = dataset_cls(data_path, language, representations, embedding_size,
                           mode, classes=classes, words=words,
                           max_instances=max_instances)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                             collate_fn=batch_generator(task))
    return trainloader, trainset.classes, trainset.words


def get_data_loaders(data_path, task, language, representations, embedding_size,
                     batch_size, max_instances=None):
    dataset_cls = get_data_cls(task)

    trainloader, classes, words = get_data_loader(
        dataset_cls, task, data_path, language, representations, embedding_size,
        'train', batch_size=batch_size, shuffle=True, max_instances=max_instances)
    devloader, classes, words = get_data_loader(
        dataset_cls, task, data_path, language, representations, embedding_size,
        'dev', batch_size=batch_size, shuffle=False, classes=classes, words=words)
    testloader, classes, words = get_data_loader(
        dataset_cls, task, data_path, language, representations, embedding_size,
        'test', batch_size=batch_size, shuffle=False, classes=classes, words=words)
    return trainloader, devloader, testloader, \
        testloader.dataset.n_classes, testloader.dataset.n_words
