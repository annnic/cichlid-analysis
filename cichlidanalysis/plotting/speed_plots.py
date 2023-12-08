import os

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import matplotlib
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.gridspec as grid_spec
import datetime as dt
from datetime import timedelta

from cichlidanalysis.plotting.single_plots import fill_plot_ts
from cichlidanalysis.utils.timings import load_timings


def plot_ridge_plots(fish_tracks_bin, change_times_datetime, rootdir, sp_metrics, tribe_col):
    """ Plot ridge plots and get averages of each feature

    :param fish_tracks_bin:
    :param change_times_datetime:
    :param rootdir:
    :param sp_metrics:
    :param tribe_col:
    :return:
    """
    # daily
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    plot_ridge_30min_combined_daily(fish_tracks_bin, feature, ymax, span_max, ylabeling, change_times_datetime, rootdir,
                                    sp_metrics, tribe_col)
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    plot_ridge_30min_combined_daily(fish_tracks_bin, feature, ymax, span_max, ylabeling, change_times_datetime, rootdir,
                                    sp_metrics, tribe_col)
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    plot_ridge_30min_combined_daily(fish_tracks_bin, feature, ymax, span_max, ylabeling, change_times_datetime, rootdir,
                                    sp_metrics, tribe_col)

    # weekly
    feature, ymax, span_max, ylabeling = 'vertical_pos', 1, 0.8, 'Vertical position'
    averages_vp, date_time_obj_vp, sp_vp_combined = plot_ridge_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    averages_spd, _, sp_spd_combined = plot_ridge_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                 ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    averages_rest, _, sp_rest_combined = plot_ridge_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                   ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    averages_move, _, sp_move_combined = plot_ridge_30min_combined(fish_tracks_bin, feature, ymax, span_max,
                                                                   ylabeling, change_times_datetime, rootdir)

    return averages_vp, date_time_obj_vp, sp_vp_combined, averages_spd, sp_spd_combined, averages_rest, \
           sp_rest_combined, averages_move, sp_move_combined


# speed_mm (30m bins) for each fish (individual lines)
def plot_speed_30m_individuals(rootdir, fish_tracks_30m, change_times_d):
    if not fish_tracks_30m.dtypes['ts'] == np.dtype('datetime64[ns]'):
        fish_tracks_30m['ts'] = pd.to_datetime(fish_tracks_30m['ts'])

    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()

    date_form = DateFormatter("%H")
    for species_f in all_species:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='speed_mm',
                          hue='FishID')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 60])
        plt.xlabel("Time (h)")
        plt.ylabel("Speed (mm/s)")
        plt.title(species_f)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 6})
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_30min_individual{0}.png".format(species_f.replace(' ', '-'))))
        plt.close()


def plot_speed_30m_sex(rootdir, fish_tracks_30m, change_times_d):
    """speed_mm (30m bins) for each fish (individual lines) coloured by sex

    """
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()

    date_form = DateFormatter("%H")
    for species_f in all_species:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='speed_mm', hue='sex',
                          units="FishID", estimator=None)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 60])
        plt.xlabel("Time (h)")
        plt.ylabel("Speed (mm/s)")
        plt.title(species_f)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 6})
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_30min_individuals_by_sex_{0}.png".format(species_f.replace(' ', '-'))))
        plt.close()

        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(data=fish_tracks_30m[fish_tracks_30m.species == species_f], x='ts', y='speed_mm', hue='sex')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 60])
        plt.xlabel("Time (h)")
        plt.ylabel("Speed (mm/s)")
        plt.title(species_f)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 6})
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_30min_mean-stdev_by_sex_{0}.png".format(species_f.replace(' ', '-'))))
        plt.close()


# speed_mm (30m bins) for each species (mean  +- std)
def plot_speed_30m_mstd(rootdir, fish_tracks_30m, change_times_d):
    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()
    date_form = DateFormatter("%H")

    for species_f in all_species:
        # get speeds for each individual for a given species
        spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # calculate ave and stdv
        average = sp_spd.mean(axis=1)
        stdv = sp_spd.std(axis=1)

        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(x=sp_spd.index, y=average + stdv, color='lightgrey')
        sns.lineplot(x=sp_spd.index, y=average - stdv, color='lightgrey')
        sns.lineplot(x=sp_spd.index, y=average)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 60])
        plt.xlabel("Time (h)")
        plt.ylabel("Speed (mm/s)")
        plt.title(species_f)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev{0}.png".format(species_f.replace(' ', '-'))))
        plt.close()


def plot_speed_30m_peaks(rootdir, fish_spd, fish_peaks_dawn, fish_peaks_dusk):
    """ Plot individual fish speed with dawn/dusk peaks """

    # fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    # day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit \
    #     = load_timings(len(fish_spd))

    date_form = DateFormatter("%H")
    date_time_obj = []
    for i in fish_spd.reset_index().ts:
        date_time_obj.append(dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
    date_time_df = pd.DataFrame(date_time_obj, columns=['ts'])

    plt.figure(figsize=(10, 4))
    ax = sns.lineplot(x=fish_spd.index, y=fish_spd)
    ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.xaxis.set_major_formatter(date_form)
    # fill_plot_ts(ax, change_times_d, date_time_df)
    days = 6
    for day in np.arange(0, days):
        ax.axvline(6 * 2 + 48*day, c='indianred')
        ax.axvline(8 * 2 + 48*day, c='indianred')
        ax.axvline(18 * 2 + 48*day, c='indianred')
        ax.axvline(20 * 2 + 48*day, c='indianred')

    ax.plot(fish_peaks_dawn[1, :], fish_peaks_dawn[2, :], "o", color="r")
    ax.plot(fish_peaks_dusk[1, :], fish_peaks_dusk[2, :], "o", color="r")
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 6*48])
    plt.xlabel("Time (h)")
    plt.ylabel("Speed (mm/s)")
    plt.title(fish_spd.name)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_peaks_{0}.png".format(fish_spd.name)))
    plt.close()
    return


def plot_ridge_30min_combined(fish_tracks_ds_i, feature, ymax, span_max, ylabeling, change_times_datetime_i, rootdir):
    """  Plot ridge plot of each species from a down sampled fish_tracks pandas structure

    :param fish_tracks_ds_i:
    :param feature:
    :param ymax:
    :param span_max:
    :param ylabeling:
    :return: averages: average speed for each
    inspiration from https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    """
    fish_IDs = fish_tracks_ds_i['FishID'].unique()
    species = fish_tracks_ds_i['species'].unique()

    cmap = cm.get_cmap('turbo')
    colour_array = np.arange(0, 1, 1 / len(species))

    date_form = DateFormatter('%H:%M:%S')

    gs = grid_spec.GridSpec(len(species), 1)
    fig = plt.figure(figsize=(16, 9))
    ax_objs = []
    averages = np.zeros([len(species), 303])

    first = 1
    for species_n, species_name in enumerate(species):
        # get speeds for each individual for a given species
        feature_i = fish_tracks_ds_i[fish_tracks_ds_i.species == species_name][[feature, 'FishID', 'ts']]
        sp_feature = feature_i.pivot(columns='FishID', values=feature, index='ts')
        if first:
            sp_feature_combined = sp_feature
            first = 0
        else:
            frames = [sp_feature_combined, sp_feature]
            sp_feature_combined = pd.concat(frames, axis=1)

        # calculate ave and stdv
        average = sp_feature.mean(axis=1)
        if np.shape(average)[0] > 303:
            averages[species_n, :] = average[0:303]
        else:
            # if data short then pad the data end with np.NaNs
            data_len = np.shape(average)[0]
            averages[species_n, :] = np.pad(average[0:data_len], (0, 303-data_len), 'constant',
                                            constant_values=(np.NaN, np.NaN))

        stdv = sp_feature.std(axis=1)
        # create time vector in datetime format
        # tv = fish_tracks_bin.loc[fish_tracks_bin.FishID == fish_IDs[0], 'ts']
        date_time_obj = []
        for i in sp_feature.index:
            date_time_obj.append(dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[species_n:species_n + 1, 0:]))

        days_to_plot = (date_time_obj[-1] - date_time_obj[0]).days + 1

        for day_n in range(days_to_plot):
            ax_objs[-1].fill_between([dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')+timedelta(days=day_n),
                                      change_times_datetime_i[0]+timedelta(days=day_n)], [span_max, span_max], 0,
                                     color='lightblue', alpha=0.5, linewidth=0, zorder=1)
            ax_objs[-1].fill_between([change_times_datetime_i[0]+timedelta(days=day_n),
                                change_times_datetime_i[1]+timedelta(days=day_n)], [span_max, span_max], 0,  color='wheat',
                                alpha=0.5, linewidth=0)
            ax_objs[-1].fill_between([change_times_datetime_i[2]+timedelta(days=day_n), change_times_datetime_i[3]+timedelta
            (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
            ax_objs[-1].fill_between([change_times_datetime_i[3]+timedelta(days=day_n), change_times_datetime_i[4]+timedelta
            (days=day_n)], [span_max, span_max], 0, color='lightblue', alpha=0.5, linewidth=0)

        # plotting the distribution
        ax_objs[-1].plot(date_time_obj, average, lw=1, color='w')
        ax_objs[-1].fill_between(date_time_obj, average, 0, color=cmap(colour_array[species_n]), zorder=2)

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-8 08:30:00", '%Y-%m-%d %H:%M:%S'))
        ax_objs[-1].set_ylim(0, ymax)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        if species_n == len(species) - 1:
            ax_objs[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
            ax_objs[-1].xaxis.set_major_locator(MultipleLocator(20))
            ax_objs[-1].xaxis.set_major_formatter(date_form)
            ax_objs[-1].yaxis.tick_right()
            ax_objs[-1].yaxis.set_label_position("right")
            ax_objs[-1].set_ylabel(ylabeling)

        else:
            # remove borders, axis ticks, and labels
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].set_xticks([])
            ax_objs[-1].set_yticks([])
            ax_objs[-1].set_yticklabels([])
            ax_objs[-1].set_ylabel('')

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(0.9, 0, species_name, fontweight="bold", fontsize=10, ha="right", rotation=-45)
        gs.update(hspace=-0.1)
    plt.savefig(os.path.join(rootdir, "{0}_30min_combined_species_{1}.png".format(feature, dt.date.today())))
    plt.close('all')
    aves_feature = pd.DataFrame(averages.T, columns=species, index=date_time_obj[0:averages.shape[1]])
    return aves_feature, date_time_obj, sp_feature_combined


def plot_ridge_30min_combined_daily(fish_tracks_ds_i, feature, ymax, span_max, ylabeling, change_times_datetime_i,
                                    rootdir, sp_metrics, tribe_col):
    """  Plot ridge plot of each species from a down sampled fish_tracks pandas structure

    :param fish_tracks_ds_i:
    :param feature:
    :param ymax:
    :param span_max:
    :param ylabeling:
    :return: averages: average speed for each
    inspiration from https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    """
    species = fish_tracks_ds_i['species'].unique()
    date_form = DateFormatter('%H:%M:%S')

    gs = grid_spec.GridSpec(len(species), 1)
    fig = plt.figure(figsize=(4, 14))
    ax_objs = []

    # order species by clustering, sort by tribe
    species_sort = fish_tracks_ds_i.loc[:, ['species', 'tribe']].drop_duplicates().sort_values('tribe').species.to_list()

    for species_n, species_name in enumerate(species_sort):
        tribe = fish_tracks_ds_i.loc[fish_tracks_ds_i["species"] == species_name].tribe.unique()[0]

        # # get speeds for each individual for a given species
        spd = fish_tracks_ds_i[fish_tracks_ds_i.species == species_name][[feature, 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values=feature, index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_spd['time_of_day'] = sp_spd.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_ave = sp_spd.groupby('time_of_day').mean()
        sp_spd_ave_std = sp_spd_ave.std(axis=1)
        daily_feature = sp_spd_ave.mean(axis=1)

        # create time vector in datetime format
        date_time_obj = []
        for i in daily_feature.index:
            date_time_obj.append(dt.datetime.strptime(i, '%H:%M')+timedelta(days=(365.25*70), hours=12))

        #  add first point at end so that there is plotting until midnight
        daily_feature = pd.concat([daily_feature, pd.Series(data=daily_feature.iloc[0], index=['24:00'])])
        date_time_obj.append(date_time_obj[-1]+timedelta(hours=0.5))

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[species_n:species_n + 1, 0:]))
        # days_to_plot = (date_time_obj[-1] - date_time_obj[0]).days + 1
        day_n = 0
        # for day_n in range(days_to_plot):
        ax_objs[-1].fill_between([dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')+timedelta(days=day_n),
                                  change_times_datetime_i[0]+timedelta(days=day_n)], [span_max, span_max], 0,
                                 color='lightblue', alpha=0.5, linewidth=0, zorder=1)
        ax_objs[-1].fill_between([change_times_datetime_i[0]+timedelta(days=day_n),
                                  change_times_datetime_i[1]+timedelta(days=day_n)], [span_max, span_max], 0,  color='wheat',
                                 alpha=0.5, linewidth=0)
        ax_objs[-1].fill_between([change_times_datetime_i[2]+timedelta(days=day_n), change_times_datetime_i[3]+timedelta
        (days=day_n)], [span_max, span_max], 0, color='wheat', alpha=0.5, linewidth=0)
        ax_objs[-1].fill_between([change_times_datetime_i[3]+timedelta(days=day_n), change_times_datetime_i[4]+timedelta
        (days=day_n)], [span_max, span_max], 0, color='lightblue', alpha=0.5, linewidth=0)

        # plotting the distribution
        ax_objs[-1].plot(date_time_obj, daily_feature, lw=1, color='w')
        ax_objs[-1].fill_between(date_time_obj, daily_feature, 0, color=tribe_col[tribe], zorder=2)

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(min(date_time_obj), dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S'))
        ax_objs[-1].set_ylim(0, ymax)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        if species_n == len(species) - 1:
            ax_objs[-1].set_xlabel("Time", fontsize=10, fontweight="bold")
            ax_objs[-1].xaxis.set_major_locator(MultipleLocator(20))
            ax_objs[-1].xaxis.set_major_formatter(date_form)
            ax_objs[-1].yaxis.tick_right()
            ax_objs[-1].yaxis.set_label_position("right")
            ax_objs[-1].set_ylabel(ylabeling)

        else:
            # remove borders, axis ticks, and labels
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].set_xticks([])
            ax_objs[-1].set_yticks([])
            ax_objs[-1].set_yticklabels([])
            ax_objs[-1].set_ylabel('')

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(1, 0, species_name, fontweight="bold", fontsize=10, ha="right", rotation=-45)
        gs.update(hspace=-0.1)
    plt.savefig(os.path.join(rootdir, "{0}_30min_combined_species_daily_{1}.png".format(feature, dt.date.today())))
    plt.close('all')
    return


def plot_speed_30m_mstd_figure(rootdir, fish_tracks_30m, change_times_d):
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()
    date_form = DateFormatter("%H")

    for species_f in all_species:
        # get speeds for each individual for a given species
        spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # calculate ave and stdv
        average = sp_spd.mean(axis=1)
        stdv = sp_spd.std(axis=1)

        plt.figure(figsize=(2, 1))
        ax = sns.lineplot(x=sp_spd.index, y=average + stdv, color='lightgrey', linewidth=0.5)
        sns.lineplot(x=sp_spd.index, y=average - stdv, color='lightgrey', linewidth=0.5)
        sns.lineplot(x=sp_spd.index, y=average, linewidth=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, 60])
        plt.xlabel("Time (h)", fontsize=SMALLEST_SIZE)
        plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)
        plt.title(species_f, fontsize=SMALLEST_SIZE)

        # Decrease the offset for tick labels on all axes
        ax.xaxis.labelpad = 0.5
        ax.yaxis.labelpad = 0.5

        # Adjust the offset for tick labels on all axes
        ax.tick_params(axis='x', pad=0.5, length=2)
        ax.tick_params(axis='y', pad=0.5, length=2)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
        ax.tick_params(width=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev_figure_{0}.pdf".format(species_f.replace(' ', '-'))), dpi=350)
        plt.close()
    return
