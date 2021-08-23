import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
from adjustText import adjust_text

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.processor import UdProcessor
from h01_data.process import get_ud_file_base
from h02_learn.dataset import get_data_loaders
from util import constants
from util import util

aspect = {
    'height': 7,
    'font_scale': 1.5,
    'labels': True,
    'name_suffix': '',
    'ratio': 1.625,
}
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 4.5
fig_size[1] = 3.5

sns.set_palette("Set2")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')

print(plt.rcParams['axes.prop_cycle'].by_key()['color'])


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--ud-path", type=str, default='data/ud/ud-treebanks-v2.6/')
    parser.add_argument("--data-path", type=str, default='data/processed/')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)

    args = parser.parse_args()

    return args


def get_pareto_points(df, x_axis, y_axis, representation, max_x):
    df.sort_values([x_axis, y_axis], ascending=[True, False], inplace=True)
    df['frontier'] = False

    max_val, old_x = -float('inf'), 0
    # max_val = 0
    for i, x in df.iterrows():
        if x[y_axis] > max_val:
            df.loc[x.name, 'frontier'] = True

            old_x = x[x_axis]
            max_val = x[y_axis]

    return df


def get_pareto_points_per_model(df, x_axis, y_axis):
    dfs = []
    max_x = df[x_axis].max()
    for rep in df.representation.unique():
        df_temp = df[df.representation == rep].copy()
        df_temp = get_pareto_points(df_temp, x_axis, y_axis, rep, max_x)
        dfs += [df_temp]

    return pd.concat(dfs)


def get_pareto_frontiers(df, x_axis, y_axis, representation, max_x):
    df.sort_values(x_axis, ascending=True, inplace=True)
    df_new = df.copy()

    old_y = 0
    for i, x in df.iterrows():
        if old_y is None:
            old_y = x[y_axis]
            continue

        x = x.copy()
        new_y = x[y_axis]
        x[y_axis] = old_y
        x[x_axis] = x[x_axis] - 0.001
        df_new = df_new.append(x)

        old_y = new_y

    x = df.iloc[-1].copy()
    x[x_axis] = max_x
    df_new = df_new.append(x)

    return df_new


def plot_pareto_frontiers(df, x_axis, y_axis, max_x, colors):
    df = df[df['frontier']].copy()
    dfs = []
    for rep in df.representation.unique():
        df_temp = df[df.representation == rep].copy()
        df_temp = get_pareto_frontiers(df_temp, x_axis, y_axis, rep, max_x)
        dfs += [df_temp]

    df = pd.concat(dfs)
    df.sort_values(y_axis, ascending=True, inplace=True)
    df.sort_values(x_axis, ascending=True, inplace=True)
    df.sort_values('Representation', ascending=True, inplace=True)

    df.reset_index(inplace=True)
    sns.lineplot(x=x_axis, y=y_axis, hue='Representation',
        data=df, legend=False, palette=colors)


def get_pareto_shade_areas(df, x_axis, y_axis):
    x_values = df[x_axis].unique()
    x_values.sort()

    shades = []

    for x_val in x_values:
        idx = df.loc[df[x_axis] <= x_val, y_axis].idxmax()
        y_val = df.loc[df[x_axis] <= x_val, y_axis].max()
        representation = df.loc[idx, 'Representation']

        shades += [[x_val, y_val, representation]]

    return shades


def repeat_shade_change(shades, max_x):
    shades_ext = [shades[0]]
    _, old_y, old_representation = shades[0]
    for i, val in enumerate(shades[1:]):
        old_val = copy.copy(val)
        old_val[1] = old_y
        old_val[2] = old_representation
        shades_ext += [old_val]
        shades_ext += [val]

        old_y = val[1]
        old_representation = val[2]

    old_val = copy.copy(val)
    old_val[0] = max_x
    old_val[1] = old_y
    old_val[2] = old_representation
    shades_ext += [old_val]

    return shades_ext


def plot_pareto_shade_areas(ax, colors, df, x_axis, y_axis, max_x):
    shades = get_pareto_shade_areas(df, x_axis, y_axis)
    shades = repeat_shade_change(shades, max_x)
    xs, ys, labels = zip(*shades)
    labels = np.array(labels)

    representations = df['Representation'].unique()
    representations.sort()

    for i, rep in enumerate(representations):
        ax.fill_between(xs, 0, ys, where=labels == rep, facecolor=colors[rep], alpha=.5)


def label_points(df, x_axis, y_axis, ax):
    max_row = df.loc[df[y_axis].idxmax()]
    min_row = df.loc[df[x_axis].idxmin()]
    df['selectivity'] = df[y_axis] - df[x_axis]
    sel_row = df.loc[df['selectivity'].idxmax()]

    data = [
        {'x': max_row[x_axis], 'y': max_row[y_axis], 'text': 'Accurate'},
        {'x': min_row[x_axis], 'y': min_row[y_axis], 'text': 'Simple'},
        {'x': sel_row[x_axis], 'y': sel_row[y_axis], 'text': 'Selective'},
    ]
    texts = []
    color = 'C7'
    for point in data:
        if point['text'] == 'Simplicity':
            texts += [ax.text(point['x'], point['y']+.01, str(point['text']), color='white', size=20, horizontalalignment='right', multialignment='right', alpha=0)]
            [ax.text(point['x']+.04, point['y'] + .06, str(point['text']), color='black', size=20, horizontalalignment='right', multialignment='right')]
        else:
            texts += [ax.text(point['x'], point['y'], str(point['text']), color='black', size=20, horizontalalignment='right', multialignment='right')]

    adjust_text(texts, df[x_axis].to_numpy(), df[y_axis].to_numpy(), arrowprops=dict(arrowstyle="->", color='black', lw=1), autoalign=False, ha='right', va='bottom')


def plot_pareto_full(df, x_axis, y_axis, colors, max_x=1, xlabel=None, ylabel='Accuracy', fname=None, add_selectivity=True):
    if xlabel is None:
        xlabel = x_axis
    df['Representation'] = df['representation']

    df = get_pareto_points_per_model(df, x_axis, y_axis)
    # df = df[df.frontier]

    df.sort_values('Representation', ascending=True, inplace=True)

    fig = plt.figure()
    plot_pareto_frontiers(df, x_axis, y_axis, max_x=max_x, colors=colors)
    ax = sns.scatterplot(
        x=x_axis, y=y_axis, hue='Representation',
        data=df, s=70, zorder=2, linewidth=0, legend=False, palette=colors)
    plot_pareto_shade_areas(ax, colors, df, x_axis, y_axis, max_x)

    min_y = df[y_axis].min()

    # plt.ylabel(ylabel)
    # plt.xlabel(xlabel)
    plt.ylabel('')
    plt.xlabel('')
    plt.ylim([-0.5, df[y_axis].max() * 1.1])
    plt.xlim([0.5, df[x_axis].max()])
    plt.xscale('log')

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_mlp(df, args, fname, colors):
    x_axis = 'shuffle_acc'
    y_axis = 'test_acc'

    df = df[~df.isna().any(1)]
    # df[x_axis] = df['ndata'].apply(np.log2)
    df[x_axis] = df['ndata']
    df[y_axis] = df['test_loss'] - df['base_test_loss']

    max_x = df[x_axis].max()

    ylabel = 'Agent-MI (bits)'
    xlabel = '# Data Examples'
    plot_pareto_full(df, x_axis, y_axis, colors, xlabel=xlabel, ylabel=ylabel, fname=fname, max_x=max_x)


def main():
    args = get_args()
    df = pd.read_csv('results/compiled_%s.tsv' % args.task, sep='\t')

    df = df[df.representation != 'onehot']
    reps = df.representation.unique()
    reps.sort()
    colors = {x: 'C' + str(i) for i, x in enumerate(reps)}
    colors['roberta'] = 'C5'

    df = df[df.language == args.language]

    fpath = 'results/plots'
    util.mkdir(fpath)
    fname = os.path.join(fpath, '%s__%s.pdf' % (args.task, args.language))

    plot_mlp(df, args, fname, colors)


if __name__ == '__main__':
    main()
