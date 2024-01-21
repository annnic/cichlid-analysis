import os

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from cichlidanalysis.analysis.processing import species_feature_fish_daily_ave


def fish_weekly_corr(rootdir, fish_tracks_ds, feature, link_method, plot=False):
    """ Finds the corr values within species across the full recording

    :param fish_tracks_ds:
    :param feature:
    :param link_method:
    :return:
    """
    species = fish_tracks_ds['species'].unique()
    first = True

    for species_i in species:
        print(species_i)
        fish_tracks_ds_sp = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, ['FishID', 'ts', feature]]
        fish_tracks_ds_sp = fish_tracks_ds_sp.pivot(columns='FishID', values=feature, index='ts')
        individ_corr = fish_tracks_ds_sp.corr()

        mask = np.ones(individ_corr.shape, dtype='bool')
        mask[np.triu_indices(len(individ_corr))] = False
        corr_val_f = individ_corr.values[mask]

        if first:
            corr_vals = pd.DataFrame(corr_val_f, columns=[species_i])
            first = False
        else:
            corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_val_f, columns=[species_i])], axis=1)

        fish_sex = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, ['FishID', 'sex']].drop_duplicates()
        fish_sex = list(fish_sex.sex)

        if plot:
            sns.clustermap(data=individ_corr, vmin=-1, vmax=1, xticklabels=fish_sex, yticklabels=fish_sex,
                             cmap='seismic')
            plt.tight_layout()
            plt.savefig(os.path.join(rootdir, "{0}_corr_by_30min_{1}_{2}.png".format(species_i, feature,
                                                                                         link_method)))
            plt.close()

    corr_vals_long = pd.melt(corr_vals, var_name='species', value_name='corr_coef')
    return corr_vals_long


def fish_daily_corr(averages_feature, feature, species_name, rootdir, link_method='single'):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :param species_name:
    :param link_method:
    :return:
    """

    # font sizes
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    # issue with some columns being all zeros and messing up correlation so drop these columns
    averages_feature_dropped = averages_feature.loc[(averages_feature.sum(axis=1) != 0), (averages_feature.sum(axis=0) != 0)]
    individ_corr = averages_feature_dropped.corr(method='pearson')
    # Z = sch.linkage(individ_corr, link_method)
    mask = np.ones(individ_corr.shape, dtype='bool')
    mask[np.triu_indices(len(individ_corr))] = False
    corr_val_f = individ_corr.values[mask]
    corr_vals = pd.DataFrame(corr_val_f, columns=[species_name])

    ax = sns.clustermap(individ_corr, figsize=(5, 2), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='RdBu_r', xticklabels=False, yticklabels=False)

    ax.fig.suptitle(feature)
    # plt.savefig(os.path.join(rootdir, "fish_of_{0}_corr_by_30min_{1}_{2}_{3}.png".format(species_name, feature, dt.date.today(), link_method)))
    plt.close()
    return corr_vals


def species_daily_corr(rootdir, averages_feature, feature, label, link_method='single'):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """

    individ_corr = averages_feature.corr()

    ax = sns.clustermap(individ_corr, figsize=(10, 9), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='viridis', yticklabels=True, xticklabels=True)
    ax.fig.suptitle(feature)
    plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}_{1}_{2}_{3}.png".format(label, feature, dt.date.today(), link_method)))
    plt.close()



def plot_corr_coefs_individual_means(rootdir, mean_corr_per_fish, feature, title):

    # font sizes
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8

    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    f, ax = plt.subplots(figsize=(2, 6))
    sns.boxplot(data=mean_corr_per_fish, x='corr_coef', y='species', ax=ax, fliersize=0, color='gainsboro',
                order=mean_corr_per_fish.groupby('species').mean().sort_values("corr_coef").index.to_list(),
                linewidth=1)
    sns.stripplot(data=mean_corr_per_fish, x='corr_coef', y='species', color=".2", ax=ax, size=2,
                  order=mean_corr_per_fish.groupby('species').mean().sort_values("corr_coef").index.to_list())
    ax.set(xlabel='Correlation', ylabel='Species')
    ax.set(xlim=(-1, 1))
    ax = plt.axvline(0, ls='--', color='k', linewidth=1)
    ax = plt.gca()
    spines = ["top", "right"]
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "intra_inidvidual_variability_fish_corr_coefs_{0}_{1}.pdf".format(feature, title)))
    plt.close()
    return


def week_corr(rootdir, fish_tracks_ds, feature, plot=False):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """
    species = fish_tracks_ds['species'].unique()

    for species_i in species:

        fishes = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, 'FishID'].unique()
        first = True
        first2 = True

        for fish in fishes:
            print(fish)
            fish_tracks_ds_day = fish_tracks_ds.loc[fish_tracks_ds.FishID == fish, ['day_n', 'time_of_day_dt', feature]]
            fish_tracks_ds_day = fish_tracks_ds_day.pivot(columns='day_n', values=feature, index='time_of_day_dt')
            individ_corr = fish_tracks_ds_day.corr()

            mask = np.ones(individ_corr.shape, dtype='bool')
            mask[np.triu_indices(len(individ_corr))] = False
            corr_val_f = individ_corr.values[mask]
            if first:
                corr_vals = pd.DataFrame(corr_val_f, columns=[fish])
                first = False
            else:
                corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_val_f, columns=[fish])], axis=1)

            if plot:
                f, ax = plt.subplots(figsize=(7, 5))
                ax = sns.clustermap(individ_corr, vmin=-1, vmax=1, cmap='bwr')
                plt.tight_layout()
                plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}.png".format(feature)))
                plt.close()

        f, ax = plt.subplots(figsize=(4, 10))
        sns.boxplot(data=corr_vals, ax=ax, fliersize=0)
        sns.stripplot(data=corr_vals, color=".2", ax=ax, size=3)
        ax.set(ylabel='Correlation')
        ax.set(ylim=(-1, 1))
        ax = plt.axhline(0, ls='--', color='k')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "individual_day_corr_coef_by_30min_{0}_{1}.png".format(feature, species_i)))
        plt.close()

        # find mean for each fish and save
        if first2:
            mean_corr_per_fish = pd.DataFrame(corr_vals.mean()).reset_index().rename(columns={'index': 'FishID', 0: "corr_coef"})
            first2 = False
        else:
            mean_corr_per_fish = pd.concat([mean_corr_per_fish, pd.DataFrame(corr_vals.mean()).reset_index().rename(columns={'index': 'FishID', 0: "corr_coef"})], axis=0)



def get_corr_coefs_daily(rootdir, fish_tracks_bin, feature, species_sixes):
    """

    :param rootdir:
    :param fish_tracks_bin:
    :param feature:
    :param species_sixes:
    :return:
    """
    first = True
    for species_name in species_sixes:
        # get daily averages for each fish in species
        fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_bin, species_name, feature)
        # correlations for individuals across average days
        corr_vals_f = fish_daily_corr(fish_daily_ave_feature, feature, species_name, rootdir)

        if first:
            corr_vals = pd.DataFrame(corr_vals_f, columns=[species_name])
            first = False
        else:
            corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_vals_f, columns=[species_name])], axis=1)

    corr_vals_long = pd.melt(corr_vals, var_name='species', value_name='corr_coef')

    return corr_vals_long


def plot_corr_coefs(rootdir, corr_vals_long, feature, title):

    # font sizes
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8

    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    f, ax = plt.subplots(figsize=(2, 6))
    sns.boxplot(data=corr_vals_long, x='corr_coef', y='species', ax=ax, fliersize=0, color='gainsboro',
                order=corr_vals_long.groupby('species').mean().sort_values("corr_coef").index.to_list(),
                linewidth=1)
    sns.stripplot(data=corr_vals_long, x='corr_coef', y='species', color=".2", ax=ax, size=2,
                  order=corr_vals_long.groupby('species').mean().sort_values("corr_coef").index.to_list())
    ax.set(xlabel='Correlation', ylabel='Species')
    ax.set(xlim=(-1, 1))
    ax = plt.axvline(0, ls='--', color='k', linewidth=1)
    ax = plt.gca()
    spines = ["top", "right"]
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "fish_corr_coefs_{0}_{1}.pdf".format(feature, title)))
    plt.close()
    return
