import os
import sys
import re
import math
# import random
import copy
import itertools
import subprocess
# import math
import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import get_data_loaders
from h02_learn.train import get_args
from util import util


def args2list(args):
    return [
        "--data-path", str(args.data_path),
        '--task', str(args.task),
        '--language', str(args.language),
        '--batch-size', str(args.batch_size),
        "--representation", str(args.representation),
        "--model", str(args.model),
        '--eval-batches', str(args.eval_batches),
        '--wait-epochs', str(args.wait_epochs),
        "--checkpoint-path", str(args.checkpoint_path),
        "--seed", str(args.seed),
    ]


def get_hyperparameters(search):
    hyperparameters = {
        '--ndata': search[0],
        '--hidden-size': search[1],
        '--nlayers': search[2],
        '--dropout': search[3],
        '--embedding-size': search[4],
    }
    return dict2list(hyperparameters)


def get_embedding_size_choices(representation):
    bert_embedding_size = list([768])
    fast_embedding_size = list([300])
    onehot_embedding_size = list({int(2**x) for x in np.arange(5.6, 8.2, 0.01)})

    if representation in ['onehot', 'random']:
        embedding_size = onehot_embedding_size
    elif representation == 'fast':
        embedding_size = fast_embedding_size
    elif representation in ['bert', 'albert', 'roberta']:
        embedding_size = bert_embedding_size
    else:
        raise ValueError('Invalid representation %s' % representation)

    return embedding_size


def loguniform_range(min_value, max_value, n_items):
    min_data, max_data = min_value, np.log2(max_value)
    ndata = np.array([
        round(2**x) for x in np.linspace(start=np.log2(min_value), stop=max_data,
                                       num=(n_items))
    ])
    while len(np.unique(ndata)) < n_items:
        min_data += n_items - len(np.unique(ndata))
        ndata = np.array(
            list(range(min_value, min_data)) +
            [round(2**x) for x in np.linspace(start=np.log2(min_data), stop=max_data,
                                            num=(n_items - min_data + min_value))]
        )

    return ndata


def get_hyperparameters_search(n_runs, representation, n_instances):
    embedding_size = get_embedding_size_choices(representation)
    hidden_size = list({int(2**x) for x in np.arange(5, 10, 0.01)})
    nlayers = [0, 1, 2]
    dropout = list(np.arange(0.0, 0.51, 0.01))
    ndata = loguniform_range(1, n_instances, n_runs)

    all_hyper = [hidden_size, nlayers, dropout, embedding_size]
    choices = []
    for hyper in all_hyper:
        choices += [np.random.choice(hyper, size=n_runs, replace=True)]
    choices = [ndata] + choices
    # print(ndata)
    # sys.exit()

    return list(zip(*choices))


def dict2list(data):
    list2d = [[k, str(x)] for k, x in data.items()]
    return list(itertools.chain.from_iterable(list2d))


def write_done(done_fname):
    with open(done_fname, "w") as f:
        f.write('done training\n')


def append_result(fname, values):
    with open(fname, "a+") as f:
        f.write(','.join(values) + '\n')


def get_results(out, err):
    res_pattern_base = r'^(\w+) (\w+). Train: (\d+.\d+) Dev: (\d+.\d+) Test: (\d+.\d+)$'

    output = out.decode().split('\n')
    results = []

    try:
        for i in range(4):
            m = re.match(res_pattern_base, output[-2 - i])
            _, res_name, train_res, dev_res, test_res = m.groups()
            results += [test_res, dev_res, train_res]

    except Exception as exc:
        print('Output:', output)
        raise ValueError('Error in subprocess: %s' % err.decode()) from exc

    return results


def run_experiment(run_count, hyper, hyperparameters, args, results_fname):
    my_env = os.environ.copy()
    cmd = ['python', 'src/h02_learn/train.py'] + args2list(args) + hyperparameters

    tqdm.write(str([args.representation] + hyperparameters))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
    out, err = process.communicate()

    results = get_results(out, err)
    append_result(results_fname,
                  [str(run_count)] + [str(x) for x in hyper] + results[::-1])


def main():
    # pylint: disable=too-many-locals

    args = get_args()
    n_runs = 50

    ouput_path = os.path.join(
        args.checkpoint_path, args.task, args.language, args.model, args.representation)
    results_fname = os.path.join(ouput_path, 'all_results.tsv')
    done_fname = os.path.join(ouput_path, 'finished.txt')

    trainloader, _, _, _, _ = \
        get_data_loaders(args.data_path, args.task, args.language,
                         'onehot', 1, 1)
    n_instances = len(trainloader.dataset)
    assert n_instances >= n_runs * 2, \
        'Should have at least as much data as instances in dataset'

    curr_iter = util.file_len(results_fname) - 1
    util.mkdir(ouput_path)

    if curr_iter == -1:
        res_columns = ['run', 'ndata',
                       'hidden_size', 'nlayers', 'dropout', 'embedding_size',
                       'train_loss', 'dev_loss', 'test_loss',
                       'train_acc', 'dev_acc', 'test_acc',
                       'base_train_loss', 'base_dev_loss', 'base_test_loss',
                       'base_train_acc', 'base_dev_acc', 'base_test_acc']
        append_result(results_fname, res_columns)
        curr_iter = 0

    search = get_hyperparameters_search(n_runs, args.representation, n_instances)

    for i, hyper in tqdm(enumerate(search[curr_iter:]), initial=curr_iter, total=n_runs):
        run_count = curr_iter + i
        hyperparameters = get_hyperparameters(hyper)

        run_experiment(run_count, hyper, hyperparameters, args, results_fname)

    write_done(done_fname)


if __name__ == '__main__':
    main()
