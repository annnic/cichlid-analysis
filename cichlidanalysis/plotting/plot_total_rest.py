import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns


def plot_total_rest_ordered(rootdir, feature_v):
    """ total rest ordered by mean

    :param rootdir:
    :param feature_v:
    :return:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})
    sns.set_context(rc={"lines.linewidth": 0.5})

    fig = plt.figure(figsize=(7, 1.5))
    ax = sns.boxplot(data=feature_v, x='six_letter_name_Ronco', y='total_rest', dodge=False,
                     showfliers=False, color='silver', linewidth=0.5,
                     order=feature_v.groupby('six_letter_name_Ronco').mean().sort_values("total_rest").index.to_list())
    for patch in ax.artists:
        fc = patch.get_facecolor()
        patch.set_facecolor(mcolors.to_rgba(fc, 0.3))
    ax = sns.swarmplot(data=feature_v, x='six_letter_name_Ronco', y='total_rest', color=".2", size=1,
                       order=feature_v.groupby('six_letter_name_Ronco').mean().sort_values(
                           "total_rest").index.to_list(), linewidth=0.5)
    ax.set(ylabel='Average total rest per day', xlabel='Species')
    ax.set(ylim=(0, 24))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.axhline(12, ls='--', color='k', lw=0.5)
    sns.despine(top=True, right=True)
    # Access the spines and set the linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)
    plt.savefig(os.path.join(rootdir, "total_rest_ordered_horizontal.pdf"), dpi=350)
    plt.close()
    return


def plot_total_rest_temporal(rootdir, feature_v):
    """ total rest ordered by mean, coloured by temporal guild

    :return:
    """
    colors = ['royalblue', 'mediumorchid', 'silver', 'gold']
    customPalette = sns.set_palette(sns.color_palette(colors))
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='total_rest', hue="cluster_pattern", dodge=False,
                     showfliers=False,
                     order=feature_v.groupby('six_letter_name_Ronco').mean().sort_values("total_rest").index.to_list())
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='total_rest', color=".2", size=4,
                       order=feature_v.groupby('six_letter_name_Ronco').mean().sort_values(
                           "total_rest").index.to_list())
    ax.set(xlabel='Average total rest per day', ylabel='Species')
    ax.set(xlim=(0, 24))
    plt.tight_layout()
    ax = plt.axvline(12, ls='--', color='k')
    plt.savefig(os.path.join(rootdir, "total_rest_ordered_cluster.png"))
    plt.close()
    return


def plot_total_rest_diet(rootdir, feature_v):
    """ total rest ordered by mean, coloured by diet guild

    :param rootdir:
    :param feature_v:
    :return:
    """
    colors = ['tomato', 'steelblue', 'sandybrown', 'mediumseagreen']
    customPalette = sns.set_palette(sns.color_palette(colors))
    fig = plt.figure(figsize=(5, 10))
    ax = sns.boxplot(data=feature_v, y='six_letter_name_Ronco', x='total_rest', hue="diet", dodge=False,
                     showfliers=False,
                     order=feature_v.groupby('six_letter_name_Ronco').mean().sort_values("total_rest").index.to_list())
    ax = sns.swarmplot(data=feature_v, y='six_letter_name_Ronco', x='total_rest', color=".2", size=4,
                       order=feature_v.groupby('six_letter_name_Ronco').mean().sort_values(
                           "total_rest").index.to_list())
    ax.set(xlabel='Average total rest per day', ylabel='Species')
    ax.set(xlim=(0, 24))
    plt.tight_layout()
    ax = plt.axvline(12, ls='--', color='k')
    plt.savefig(os.path.join(rootdir, "total_rest_ordered_diet.png"))
    plt.close()
    return

def plot_total_rest_hist(rootdir, feature_v, feature_v_mean):
    """ histogram of total rest timings
    :param rootdir:
    :param feature_v:
    :return:
    """

    fig = plt.figure(figsize=(10, 5))
    sns.histplot(data=feature_v, x='total_rest', binwidth=1, multiple="stack", color='skyblue').set(
        title='Total rest per fish')
    plt.savefig(os.path.join(rootdir, "total_rest_hist_per_fish.png"))
    plt.close()
    sns.histplot(data=feature_v_mean, x='total_rest', binwidth=1, multiple="stack", color='royalblue').set(
        title='Total rest per species')
    plt.savefig(os.path.join(rootdir, "total_rest_hist_per_species.png"))
    plt.close()
    return


def plot_total_rest_vs_spd(rootdir, feature_v):
    # total rest vs day speed
    feature_v['spd_max_mean'] = pd.concat([feature_v['spd_mean_day'], feature_v['spd_mean_night'],
                                           feature_v['spd_mean_predawn'], feature_v['spd_mean_dawn'],
                                           feature_v['spd_mean_dusk'], feature_v['spd_mean_postdusk']], axis=1).max(
        axis=1)

    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(data=feature_v, x='total_rest', y='spd_mean_day')
    plt.savefig(os.path.join(rootdir, "total_rest_vs_spd_mean_day.png"))
    plt.close()
    return

