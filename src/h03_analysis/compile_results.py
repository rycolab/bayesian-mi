import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.processor import UdProcessor
from h01_data.process import get_ud_file_base
from h02_learn.dataset import get_data_loaders
from util import constants
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--ud-path", type=str, default='data/ud/ud-treebanks-v2.6/')
    parser.add_argument("--data-path", type=str, default='data/processed/')
    parser.add_argument('--task', type=str, required=True)
    # Others
    parser.add_argument("--checkpoint-path", type=str, default='checkpoints/')

    args = parser.parse_args()

    return args


def get_representation_results(checkpoint_path, task, language, model, representation):
    fname = os.path.join(checkpoint_path, task, language, model, representation, 'all_results.tsv')
    df = pd.read_csv(fname)
    df['representation'] = representation

    return df


def get_lang_results(checkpoint_path, task, lang):
    dfs = []
    if lang == 'english':
        representations = ['bert', 'fast', 'random', 'albert', 'roberta']
    else:
        representations = ['bert', 'fast', 'random']

    for model in ['mlp']:
        for rep in representations:
            try:
                df_rep = get_representation_results(checkpoint_path, task, lang, model, rep)
                df_rep['model'] = model
                dfs += [df_rep]
            except FileNotFoundError:
                print('\tSkipping language with model %s and representation %s' % (model, rep))

    if not dfs:
        return None

    df = pd.concat(dfs)
    df['language'] = lang

    return df


def get_raw_ud(ud_path, language):
    mode = 'train'
    ud_file_base = get_ud_file_base(ud_path, language)
    ud_file = ud_file_base % mode
    processor = UdProcessor()
    ud_data = processor.tokenize(ud_file)
    return ud_data


def get_entropy(df_ud):
    df_ud['count'] = 1
    df = df_ud[['count', 'label']].groupby('label').agg('count')
    df['probs'] = df['count'] / df['count'].sum()
    df['logprobs'] = np.log2(df['probs'])

    entropy = - (df.probs * df.logprobs).sum()
    return entropy


def get_word_counts(df, df_ud):
    df_ud['count_word'] = 1
    df_word = df_ud[['count_word', 'word']].groupby(['word']).agg('count')
    df = df_word.join(df.set_index('word'), how='outer')
    return df.reset_index()


def get_entropy_condition_words(df_ud):
    df_ud['count'] = 1
    df = df_ud[['count', 'word', 'label']].groupby(['word', 'label']).agg('count').reset_index()

    df = get_word_counts(df, df_ud)
    assert (df['count_word'] != df['count']).any()

    df['probs'] = df['count'] / df['count_word']
    df['surprisal'] = - df['probs'] * np.log2(df['probs'])
    df_word = df[['surprisal', 'word']].groupby(['word']).agg('sum').reset_index()
    df_word = get_word_counts(df_word, df_ud)

    df_word['probs'] = df_word['count_word'] / df_word['count_word'].sum()
    entropy = (df_word['probs'] * df_word['surprisal']).sum()

    return entropy


def get_task_column(task, df_ud):
    if task == 'pos_tag':
        df_ud = df_ud[(df_ud.pos != 'X') & (df_ud.pos != '_')]
        return 'pos', df_ud
    if task == 'dep_label':
        df_ud = df_ud[(df_ud.rel != 'root') & (df_ud.rel != '_')]
        return 'rel', df_ud
    raise ValueError('Invalid task: %s' % task)


def get_lang_entropy(df_ud):
    entropy = get_entropy(df_ud)
    entropy_conditional = get_entropy_condition_words(df_ud)

    return entropy, entropy_conditional


def ud_to_dataframe(dataloader, task):
    if task == 'pos_tag':
        ud_data = [
            {'word': x.item(), 'label': y.item()}
            for (batch_x, batch_y) in dataloader
            for x, y in zip(batch_x, batch_y)
        ]
    elif task == 'dep_label':
        ud_data = [
            {'word': x[0].item(), 'head': x[1].item(), 'label': y.item()}
            for (batch_x, batch_y) in dataloader
            for x, y in zip(batch_x, batch_y)
        ]
    return pd.DataFrame(ud_data)


def get_ud_data(data_path, task, language):
    representation = 'onehot'
    embedding_size = 1
    batch_size = 64

    trainloader, devloader, testloader, n_classes, _ = \
        get_data_loaders(data_path, task, language, representation, embedding_size, batch_size)

    df_train = ud_to_dataframe(trainloader, task)
    df_dev = ud_to_dataframe(devloader, task)
    df_test = ud_to_dataframe(testloader, task)

    return df_train, df_dev, df_test, n_classes


def append_lang_stats(df, data_path, task, language):

    df_train, df_dev, df_test, n_classes = get_ud_data(data_path, task, language)
    df_ud = pd.concat([df_train, df_dev, df_test])

    entropy, entropy_conditional = get_lang_entropy(df_ud)
    n_classes = df_ud['label'].unique().shape[0]
    print(language, entropy, entropy_conditional)

    df['n_train'] = df_train.shape[0]
    df['n_test'] = df_test.shape[0]
    df['entropy'] = entropy
    df['entropy_conditional'] = entropy_conditional
    df['n_classes'] = n_classes
    return df


def get_lang_info(data_path, checkpoint_path, task, lang):
    df = get_lang_results(checkpoint_path, task, lang)
    if df is not None and task != 'parse':
        df = append_lang_stats(df, data_path, task, lang)
    return df


def main():
    args = get_args()

    dfs = []
    for lang in constants.USED_LANGUAGES:
        print(lang)
        df_lang = get_lang_info(
            args.data_path, args.checkpoint_path, args.task, lang)

        if df_lang is not None:
            dfs += [df_lang]

    df = pd.concat(dfs)

    util.mkdir('results/')
    df.to_csv('results/compiled_%s.tsv' % args.task, sep='\t')


if __name__ == '__main__':
    main()
