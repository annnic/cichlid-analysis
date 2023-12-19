import os

import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from matplotlib.dates import DateFormatter
from datetime import timedelta

from cichlidanalysis.utils.timings import output_timings


def get_keys_by_val(dict, value_to_find):
    listOfKeys = list()
    listOfItems = dict.items()
    for item in listOfItems:
        if item[1].count(value_to_find) > 0:
            listOfKeys.append(item[0])
    return listOfKeys


def cluster_dics():
    # dic_complex = {'diurnal': [6], 'nocturnal': [1], 'crepuscular1': [2], 'crepuscular2': [4], 'crepuscular3': [5],
    #        'undefined': [3, 7, 8, 9, 10, 11]}
    # dic_simple = {'diurnal': [6], 'nocturnal': [1], 'crepuscular': [2, 4, 5], 'undefined': [3, 7, 8, 9, 10, 11]}

    # final
    dic_complex = {'diurnal': [7], 'nocturnal': [1], 'crepuscular1': [2], 'crepuscular2': [4], 'crepuscular3': [6],
           'crepuscular4': [5], 'undefined': [3, 8, 9, 10, 11, 12]}
    dic_simple = {'diurnal': [7], 'nocturnal': [1], 'crepuscular': [2, 4, 5, 6], 'undefined': [3, 8, 9, 10, 11, 12]}

    col_dic_complex = {'diurnal': 'orange', 'nocturnal': 'royalblue', 'crepuscular1': 'orchid', 'crepuscular2':
        'mediumorchid', 'crepuscular3': 'darkorchid',  'crepuscular4': 'mediumpurple', 'undefined': 'dimgrey'}
    col_dic_simple = {'diurnal': 'orange', 'nocturnal': 'royalblue', 'crepuscular': 'orchid', 'undefined': 'dimgrey'}

    # cluster_dic = {'1': 'nocturnal', '2': 'crepuscular1', '3': 'undefined', '4': 'crepuscular2', '5': 'crepuscular3',
    #                '6': 'diurnal', '7': 'undefined', '8': 'undefined', '9': 'undefined', '10': 'undefined',
    #                '11': 'undefined'}
    cluster_order = [12, 11, 10, 9, 3, 1, 2, 4, 6, 5, 8, 7]
    return dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order


def dendrogram_sp_clustering(aves_ave_spd, link_method='single', max_d=1.35):
    """ Dendrogram of the clustering as done in clustermap. This allows me to get out the clusters

    :param aves_ave_spd:
    :param link_method:
    :param max_d:
    :return:
    """
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    aves_ave_spd = aves_ave_spd.reindex(sorted(aves_ave_spd.columns), axis=1)
    individ_corr = aves_ave_spd.corr(method='pearson')
    z = linkage(individ_corr, link_method)

    plt.figure(figsize=[8, 5])
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        labels=individ_corr.index,
        color_threshold=max_d
    )
    plt.axhline(max_d, color='k')
    plt.close()

    clusters = fcluster(z, max_d, criterion='distance')
    d = {'species_six': individ_corr.index, "cluster": clusters}
    species_cluster = pd.DataFrame(d)
    species_cluster = species_cluster.sort_values(by="cluster")
    species_cluster['colour'] = 'grey'
    for col in col_dic_complex:
        cluster_n = dic_complex[col]
        species_cluster.loc[species_cluster.cluster.isin(cluster_n), 'colour'] = col_dic_complex[col]

    return individ_corr, species_cluster


def clustered_spd_map(rootdir, aves_ave_spd, link_method='single', max_d=1.35):

    individ_corr, species_cluster = dendrogram_sp_clustering(aves_ave_spd, link_method=link_method, max_d=max_d)

    # Plot cluster map with one dendrogram, main clusters (hardcoded) as row/col colours
    cg = sns.clustermap(individ_corr, figsize=(12, 12), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='viridis', yticklabels=True, xticklabels=True,
                        row_colors=species_cluster.sort_values(by='species_six').colour.to_list(),
                        col_colors=species_cluster.sort_values(by='species_six').colour.to_list(),
                        cbar_kws={'label': 'Correlation coefficient'})
    cg.ax_col_dendrogram.set_visible(False)
    plt.savefig(os.path.join(rootdir, "figure_panel_1_clustermap_{}.png".format(dt.date.today())))
    plt.close()
    return


def cluster_daily_ave(rootdir, aves_ave_spd, label, link_method='single', max_d=1.35):
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, \
    change_times_datetime, change_times_unit = output_timings()
    individ_corr, species_cluster = dendrogram_sp_clustering(aves_ave_spd, link_method=link_method, max_d=max_d)

    date_form = DateFormatter('%H')
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, len(cluster_order)*25, 'Speed mm/s'
    fig = plt.figure(figsize=(2, 9))

    # create time vector in datetime format
    date_time_obj = []
    for i in aves_ave_spd.index:
        date_time_obj.append(dt.datetime.strptime(i, '%H:%M') + timedelta(days=(365.25 * 70), hours=12))

    day_n = 0
    plt.fill_between(
        [dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S') + timedelta(days=day_n),
         change_times_datetime[0] + timedelta(days=day_n)], [span_max, span_max], 0,
        color='lightblue', alpha=0.5, linewidth=0, zorder=1)
    plt.fill_between([change_times_datetime[0] + timedelta(days=day_n),
                      change_times_datetime[1] + timedelta(days=day_n)], [span_max, span_max], 0,
                     color='wheat',
                     alpha=0.5, linewidth=0)
    plt.fill_between([change_times_datetime[2] + timedelta(days=day_n), change_times_datetime[3] + timedelta
    (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
    plt.fill_between([change_times_datetime[3] + timedelta(days=day_n), change_times_datetime[4] + timedelta
    (days=day_n)], [span_max, span_max], 0, color='lightblue', alpha=0.5, linewidth=0)

    top = len(cluster_order) * 25
    for cluster_count, cluster_n in enumerate(cluster_order):
        subset_spe = species_cluster.loc[species_cluster.cluster == cluster_n, 'species_six']
        subset_spd = aves_ave_spd.loc[:, aves_ave_spd.columns.isin(subset_spe)]
        # subset_spd_stdev = subset_spd.std(axis=1)
        daily_speed = subset_spd.mean(axis=1) + top - 25 - cluster_count * 25

        colour = col_dic_complex[get_keys_by_val(dic_complex, cluster_n)[0]]
        # plotting
        # ax = sns.lineplot(x=date_time_obj, y=(daily_speed + subset_spd_stdev), color='lightgrey')
        # ax = sns.lineplot(x=date_time_obj, y=(daily_speed - subset_spd_stdev), color='lightgrey')
        for species in subset_spe:
            ax = sns.lineplot(x=date_time_obj, y=subset_spd.loc[:, species] + top - 25 - cluster_count * 25,
                              color=colour, alpha=0.3)
        ax = sns.lineplot(x=date_time_obj, y=daily_speed, color=colour, linewidth=3)

    # setting uniform x and y lims
    ax = plt.gca()
    ax.set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S'))
    ax.set_ylim(0, span_max)

    ax.set_xlabel("Time", fontsize=10) #, fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(date_form)

    ax.yaxis.set_ticks(np.arange(0, 30, step=10))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(ylabeling, loc='bottom')

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "figure_panel_1_daily_traces_{}.png".format(label)))
    plt.close()
    return


def plot_all_spd_subplots(rootdir, fish_tracks_bin, change_times_datetime, loadings):
    date_form = DateFormatter('%H')
    fish_IDs = fish_tracks_bin['FishID'].unique()
    species = fish_tracks_bin['species'].unique()
    feature = 'speed_mm'
    span_max = 100
    day_n = 0

    # font sizes
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    sorted_loadings = loadings.sort_values(by='pc1')
    data_minmax = sorted_loadings.pc1
    if data_minmax.min() < 0:
        end_val = np.max([abs(data_minmax.max()), abs(data_minmax.min())])
        df_scaled = (data_minmax + end_val) / (end_val + end_val)
    else:
        print('need to check scaling')
    rows = 5
    cols = 12
    n_plots = rows*cols

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 5))
    # Flatten the 2D array of subplots to make it easier to iterate
    axes = axes.flatten()

    for species_n, species_name in enumerate(sorted_loadings.species):
        # get speeds for each individual for a given species
        feature_i = fish_tracks_bin[fish_tracks_bin.species == species_name][[feature, 'FishID', 'ts']]
        sp_feature = feature_i.pivot(columns='FishID', values=feature, index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_feature['time_of_day'] = sp_feature.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_ave = sp_feature.groupby('time_of_day').mean()
        sp_spd_ave_std = sp_spd_ave.std(axis=1)
        daily_feature = sp_spd_ave.mean(axis=1)

        # make datetime consistent, also make the points the middle of the bin
        time_dif = dt.datetime.strptime("1970-1-1 23:45:00", '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime('00:00', '%H:%M')
        date_time_obj = []
        for i in daily_feature.index:
            date_time_obj.append(dt.datetime.strptime(i, '%H:%M')+time_dif)

        # for day_n in range(days_to_plot):
        night_col = 'lightblue'
        axes[species_n].fill_between(
            [dt.datetime.strptime("1970-1-1 23:30:00", '%Y-%m-%d %H:%M:%S') + timedelta(days=day_n),
             change_times_datetime[0] + timedelta(days=day_n)], [span_max, span_max], 0,
            color=night_col, alpha=0.5, linewidth=0, zorder=1)
        axes[species_n].fill_between([change_times_datetime[0] + timedelta(days=day_n),
                                  change_times_datetime[1] + timedelta(days=day_n)], [span_max, span_max], 0,
                                 color='wheat',
                                 alpha=0.5, linewidth=0)
        axes[species_n].fill_between(
            [change_times_datetime[2] + timedelta(days=day_n), change_times_datetime[3] + timedelta
            (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
        axes[species_n].fill_between(
            [change_times_datetime[3] + timedelta(days=day_n), change_times_datetime[4] + timedelta
            (days=day_n)], [span_max, span_max], 0, color=night_col, alpha=0.5, linewidth=0)

        # plot speed data
        axes[species_n].plot(date_time_obj, (daily_feature + sp_spd_ave_std), lw=1, color='lightgrey')
        axes[species_n].plot(date_time_obj, (daily_feature - sp_spd_ave_std), lw=1, color='lightgrey')
        # cmap = plt.get_cmap('RdBu')
        # cmap(df_scaled.iloc[species_n])
        axes[species_n].plot(date_time_obj, daily_feature, lw=1, color='#1f77b4')
        axes[species_n].set_title(species_name, y=0.85, fontsize=MEDIUM_SIZE)

        if species_n % cols == 0 and species_n >= cols*(rows-1):
            axes[species_n].set_xlabel("Time (h)", fontsize=MEDIUM_SIZE)
            axes[species_n].xaxis.set_major_locator(MultipleLocator(0.25))
            axes[species_n].xaxis.set_major_formatter(date_form)
            axes[species_n].set_yticks([0, 25, 50, 75])
            axes[species_n].tick_params(axis='y', labelsize=MEDIUM_SIZE)
            axes[species_n].set_ylabel('Speed mm/s', fontsize=MEDIUM_SIZE)
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            for axis in ['bottom', 'left']:
                axes[species_n].spines[axis].set_linewidth(0.5)
            axes[species_n].tick_params(width=0.5)
            plt.setp(axes[species_n].xaxis.get_majorticklabels(), rotation=70)
        elif species_n % cols == 0:

            axes[species_n].set_yticks([0, 25, 50, 75])
            axes[species_n].set_xticks([])
            axes[species_n].tick_params(axis='y', labelsize=MEDIUM_SIZE)
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            axes[species_n].spines['bottom'].set_visible(False)
            axes[species_n].spines['left'].set_linewidth(0.5)
            axes[species_n].tick_params(width=0.5)
        elif species_n > cols*(rows-1):
            axes[species_n].set_yticks([])
            axes[species_n].xaxis.set_major_locator(MultipleLocator(0.25))
            axes[species_n].xaxis.set_major_formatter(date_form)
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            axes[species_n].spines['left'].set_visible(False)
            axes[species_n].spines['bottom'].set_linewidth(0.5)
            axes[species_n].tick_params(width=0.5)
            plt.setp(axes[species_n].xaxis.get_majorticklabels(), rotation=70)
        else:
            # remove borders, axis ticks, and labels
            axes[species_n].set_xticklabels([])
            axes[species_n].set_xticks([])
            axes[species_n].set_yticks([])
            axes[species_n].set_yticklabels([])
            axes[species_n].set_ylabel('')
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            axes[species_n].spines['bottom'].set_visible(False)
            axes[species_n].spines['left'].set_visible(False)
        axes[species_n].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S'))
        axes[species_n].set_ylim(0, span_max)
    for empty_plots in np.arange(n_plots-(n_plots-len(sorted_loadings.species)), n_plots):
        axes[empty_plots].set_xticklabels([])
        axes[empty_plots].set_xticks([])
        axes[empty_plots].set_yticks([])
        axes[empty_plots].set_yticklabels([])
        axes[empty_plots].set_ylabel('')
        axes[empty_plots].spines['top'].set_visible(False)
        axes[empty_plots].spines['right'].set_visible(False)
        axes[empty_plots].spines['bottom'].set_visible(False)
        axes[empty_plots].spines['left'].set_visible(False)

    # want to add cmap
    # cax = axes[empty_plots-1].scatter(data_minmax, data_minmax, cmap=cmap)
    # fig.colorbar(cax, ax=axes[empty_plots], orientation='vertical')
    plt.savefig(os.path.join(rootdir, 'speed_30min_ave_ave-stdev_all.pdf'), format='pdf', dpi=350)
    plt.close()
    return


def plot_all_spd_zscore_subplots(rootdir, fish_tracks_bin, change_times_datetime, loadings):
    date_form = DateFormatter('%H')
    fish_IDs = fish_tracks_bin['FishID'].unique()
    species = fish_tracks_bin['species'].unique()
    feature = 'speed_mm'
    span_max = 100
    day_n = 0

    # font sizes
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    sorted_loadings = loadings.sort_values(by='pc1')
    data_minmax = sorted_loadings.pc1
    if data_minmax.min() < 0:
        end_val = np.max([abs(data_minmax.max()), abs(data_minmax.min())])
        df_scaled = (data_minmax + end_val) / (end_val + end_val)
    else:
        print('need to check scaling')
    rows = 5
    cols = 12
    n_plots = rows*cols

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 5))
    # Flatten the 2D array of subplots to make it easier to iterate
    axes = axes.flatten()

    for species_n, species_name in enumerate(sorted_loadings.species):
        # get speeds for each individual for a given species
        feature_i = fish_tracks_bin[fish_tracks_bin.species == species_name][[feature, 'FishID', 'ts']]
        sp_feature = feature_i.pivot(columns='FishID', values=feature, index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_feature['time_of_day'] = sp_feature.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_ave = sp_feature.groupby('time_of_day').mean()
        sp_spd_ave_std = sp_spd_ave.std(axis=1)
        daily_feature = sp_spd_ave.mean(axis=1)

        # make datetime consistent, also make the points the middle of the bin
        time_dif = dt.datetime.strptime("1970-1-1 23:45:00", '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime('00:00', '%H:%M')
        date_time_obj = []
        for i in daily_feature.index:
            date_time_obj.append(dt.datetime.strptime(i, '%H:%M')+time_dif)

        # for day_n in range(days_to_plot):
        night_col = 'lightblue'
        axes[species_n].fill_between(
            [dt.datetime.strptime("1970-1-1 23:30:00", '%Y-%m-%d %H:%M:%S') + timedelta(days=day_n),
             change_times_datetime[0] + timedelta(days=day_n)], [span_max, span_max], 0,
            color=night_col, alpha=0.5, linewidth=0, zorder=1)
        axes[species_n].fill_between([change_times_datetime[0] + timedelta(days=day_n),
                                  change_times_datetime[1] + timedelta(days=day_n)], [span_max, span_max], 0,
                                 color='wheat',
                                 alpha=0.5, linewidth=0)
        axes[species_n].fill_between(
            [change_times_datetime[2] + timedelta(days=day_n), change_times_datetime[3] + timedelta
            (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
        axes[species_n].fill_between(
            [change_times_datetime[3] + timedelta(days=day_n), change_times_datetime[4] + timedelta
            (days=day_n)], [span_max, span_max], 0, color=night_col, alpha=0.5, linewidth=0)

        # plot speed data
        axes[species_n].plot(date_time_obj, (daily_feature + sp_spd_ave_std), lw=1, color='lightgrey')
        axes[species_n].plot(date_time_obj, (daily_feature - sp_spd_ave_std), lw=1, color='lightgrey')
        # cmap = plt.get_cmap('RdBu')
        # cmap(df_scaled.iloc[species_n])
        axes[species_n].plot(date_time_obj, daily_feature, lw=1, color='#1f77b4')
        axes[species_n].set_title(species_name, y=0.85, fontsize=MEDIUM_SIZE)

        if species_n % cols == 0 and species_n >= cols*(rows-1):
            axes[species_n].set_xlabel("Time (h)", fontsize=MEDIUM_SIZE)
            axes[species_n].xaxis.set_major_locator(MultipleLocator(0.25))
            axes[species_n].xaxis.set_major_formatter(date_form)
            axes[species_n].set_yticks([0, 25, 50, 75])
            axes[species_n].tick_params(axis='y', labelsize=MEDIUM_SIZE)
            axes[species_n].set_ylabel('Speed mm/s', fontsize=MEDIUM_SIZE)
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            for axis in ['bottom', 'left']:
                axes[species_n].spines[axis].set_linewidth(0.5)
            axes[species_n].tick_params(width=0.5)
            plt.setp(axes[species_n].xaxis.get_majorticklabels(), rotation=70)
        elif species_n % cols == 0:

            axes[species_n].set_yticks([0, 25, 50, 75])
            axes[species_n].set_xticks([])
            axes[species_n].tick_params(axis='y', labelsize=MEDIUM_SIZE)
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            axes[species_n].spines['bottom'].set_visible(False)
            axes[species_n].spines['left'].set_linewidth(0.5)
            axes[species_n].tick_params(width=0.5)
        elif species_n > cols*(rows-1):
            axes[species_n].set_yticks([])
            axes[species_n].xaxis.set_major_locator(MultipleLocator(0.25))
            axes[species_n].xaxis.set_major_formatter(date_form)
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            axes[species_n].spines['left'].set_visible(False)
            axes[species_n].spines['bottom'].set_linewidth(0.5)
            axes[species_n].tick_params(width=0.5)
            plt.setp(axes[species_n].xaxis.get_majorticklabels(), rotation=70)
        else:
            # remove borders, axis ticks, and labels
            axes[species_n].set_xticklabels([])
            axes[species_n].set_xticks([])
            axes[species_n].set_yticks([])
            axes[species_n].set_yticklabels([])
            axes[species_n].set_ylabel('')
            axes[species_n].spines['top'].set_visible(False)
            axes[species_n].spines['right'].set_visible(False)
            axes[species_n].spines['bottom'].set_visible(False)
            axes[species_n].spines['left'].set_visible(False)
        axes[species_n].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S'))
        axes[species_n].set_ylim(0, span_max)
    for empty_plots in np.arange(n_plots-(n_plots-len(sorted_loadings.species)), n_plots):
        axes[empty_plots].set_xticklabels([])
        axes[empty_plots].set_xticks([])
        axes[empty_plots].set_yticks([])
        axes[empty_plots].set_yticklabels([])
        axes[empty_plots].set_ylabel('')
        axes[empty_plots].spines['top'].set_visible(False)
        axes[empty_plots].spines['right'].set_visible(False)
        axes[empty_plots].spines['bottom'].set_visible(False)
        axes[empty_plots].spines['left'].set_visible(False)

    plt.savefig(os.path.join(rootdir, 'speed_30min_zscore_ave_ave-stdev_all.pdf'), format='pdf', dpi=350)
    plt.close()
    return
