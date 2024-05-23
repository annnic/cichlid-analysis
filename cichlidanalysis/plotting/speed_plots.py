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
import matplotlib.patches as patches

from cichlidanalysis.plotting.single_plots import fill_plot_ts
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.crepuscular_pattern import peak_borders


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
    plt.savefig(os.path.join(rootdir, "{0}_30min_combined_species.png".format(feature)))
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
    plt.savefig(os.path.join(rootdir, "{0}_30min_combined_species_daily.png".format(feature)))
    plt.close('all')
    return


def plot_speed_30m_mstd_figure(rootdir, fish_tracks_30m, change_times_d, ylim_max=60):
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
        ax.set_ylim([0, ylim_max])
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
        plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev_figure_{0}_ylim_[].pdf".format(species_f.replace(' ', '-'), ylim_max)), dpi=350)
        plt.close()
    return


def plot_speed_30m_mstd_figure_info(rootdir, fish_tracks_30m, change_times_d, diel_guilds, cichlid_meta, temporal_col,
                                    ylim_max=60):
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    # get each species
    all_species = fish_tracks_30m['species'].unique()
    # get each fish ID
    fish_IDs = fish_tracks_30m['FishID'].unique()
    date_form = DateFormatter("%H")

    temporal_colors = diel_guilds.diel_guild.map(temporal_col)
    temporal_colors = temporal_colors.set_axis(diel_guilds.species)

    rect_x = 0  # x-coordinate of the bottom-left corner of the rectangle
    rect_y = ylim_max  # y-coordinate of the bottom-left corner of the rectangle
    rect_width = 20  # width of the rectangle
    rect_height = 2  # height of the rectangle

    for species_f in all_species:
        # Note had to manually combine: Julmrk: ['Julidochromis marksmithi', 'Julidochromis regani']
        # spd = fish_tracks_30m.loc[fish_tracks_30m.species.isin(['Julidochromis marksmithi', 'Julidochromis regani']), ['speed_mm', 'FishID', 'ts']]
        # get speeds for each individual for a given species
        spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # calculate ave and stdv
        average = sp_spd.mean(axis=1)
        stdv = sp_spd.std(axis=1)

        # plt.figure(figsize=(2, 1))
        plt.figure(figsize=(1.6, 0.9))
        ax = sns.lineplot(x=sp_spd.index, y=average + stdv, color='lightgrey', linewidth=0.5)
        sns.lineplot(x=sp_spd.index, y=average - stdv, color='lightgrey', linewidth=0.5)
        sns.lineplot(x=sp_spd.index, y=average, linewidth=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        fill_plot_ts(ax, change_times_d, fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[0]].ts)
        ax.set_ylim([0, ylim_max])
        plt.xlabel("Time (h)", fontsize=SMALLEST_SIZE)
        plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)

        # get all names
        sp_meta = cichlid_meta.loc[cichlid_meta.species_our_names == species_f, :]
        plt.title(sp_meta.species_true.values[0] + ' (' + sp_meta.six_letter_name_Ronco.values[0] + ')', fontsize=SMALLEST_SIZE)

        # add N number
        species_num = spd.FishID.unique().shape[0]
        ax.text(1.1, 70, species_num)

        # add temporal guild coloured rectangle
        temporal_c = temporal_colors.loc[temporal_colors.index == sp_meta.six_letter_name_Ronco.values[0]][0]
        # rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=0.5, edgecolor='none',
        #                               facecolor=temporal_c)
        rectangle = patches.Rectangle((rect_x, rect_y-rect_height-0.1), rect_width, rect_height, linewidth=0.5, edgecolor='none',
                                      facecolor=temporal_c)
        ax.add_patch(rectangle)

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
        plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev_figure_{0}_ylim_{1}_N_and_temporal.pdf".format(species_f.replace(' ', '-'), ylim_max)), dpi=350)
        plt.close()
    return

def plot_speed_30m_mstd_figure_light_perturb(rootdir, fish_tracks_30m, change_times_d, normal_days=4):
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
        tv_internal = fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[1]].ts

        td = tv_internal.iloc[-1] - tv_internal.iloc[0]
        days = td.round('d')
        if td > days:
            days = days + '1d'
        days_to_plot = days.days + 1

        # for the normal days, plot the different light conditions
        for day_n in range(normal_days):
            ax.axvspan(0 + day_n, change_times_d[0] + day_n, color='lightblue', alpha=0.5, linewidth=0)
            ax.axvspan(change_times_d[0] + day_n, change_times_d[1] + day_n, color='wheat', alpha=0.5,
                       linewidth=0)
            ax.axvspan(change_times_d[2] + day_n, change_times_d[3] + day_n, color='wheat', alpha=0.5,
                       linewidth=0)
            ax.axvspan(change_times_d[3] + day_n, day_n + 1, color='lightblue', alpha=0.5, linewidth=0)

        # for the rest of the days, plot Dark:Dark
        for day_n in np.arange(normal_days, days_to_plot):
            ax.axvspan(0 + day_n, change_times_d[0] + day_n, color='lightblue', alpha=0.5, linewidth=0)
            ax.axvspan(change_times_d[0] + day_n, change_times_d[3] + day_n, color='cadetblue', alpha=0.3,
                       linewidth=0)
            ax.axvspan(change_times_d[3] + day_n, day_n + 1, color='lightblue', alpha=0.5, linewidth=0)
        # ax.axvspan(day_n + 1, days_to_plot, color='lightblue', alpha=0.5, linewidth=0)


        ax.set_ylim([0, 100])
        td = tv_internal.iloc[-1] - tv_internal.iloc[0]
        days = td.round('d')
        if td > days:
            days = days + '1d'
        days_to_plot = days.days + 1
        ax.set_xlim([1, days_to_plot - 16 / 24])
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
        plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev_figure_light_{0}.pdf".format(species_f.replace(' ', '-'))), dpi=350)
        plt.close()
    return


def plot_speed_30m_mstd_figure_conditions(rootdir, fish_tracks_30m, change_times_d, tag1, tag2, measure_epochs):
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
        spd = fish_tracks_30m[fish_tracks_30m.species == species_f][['speed_mm', 'FishID', 'ts', 'condition']]
        spd_tag1 = spd.loc[spd.condition == tag1]
        spd_tag2 = spd.loc[spd.condition == tag2]
        sp_spd_tag1 = spd_tag1.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_tag2 = spd_tag2.pivot(columns='FishID', values='speed_mm', index='ts')

        # calculate ave and stdv
        average_tag1 = sp_spd_tag1.mean(axis=1)
        stdv_tag1 = sp_spd_tag1.std(axis=1)

        average_tag2 = sp_spd_tag2.mean(axis=1)
        stdv_tag2 = sp_spd_tag2.std(axis=1)

        plt.figure(figsize=(2, 1))
        ax = sns.lineplot(x=average_tag1.index, y=average_tag1 + stdv_tag1, linewidth=0.5, color='m', alpha=0.2)
        sns.lineplot(x=average_tag1.index, y=average_tag1 - stdv_tag1, linewidth=0.5, color='m', alpha=0.2)

        ax = sns.lineplot(x=average_tag2.index, y=average_tag2 + stdv_tag2, linewidth=0.5, color='k', alpha=0.2)
        sns.lineplot(x=average_tag2.index, y=average_tag2 - stdv_tag2, linewidth=0.5, color='k', alpha=0.2)

        sns.lineplot(x=average_tag1.index, y=average_tag1, linewidth=0.5, color='m')
        sns.lineplot(x=average_tag2.index, y=average_tag2, linewidth=0.5, color='k')

        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(date_form)
        tv_internal = fish_tracks_30m[fish_tracks_30m.FishID == fish_IDs[1]].ts
        fill_plot_ts(ax, change_times_d, tv_internal)

        # add day injection timing on 4th and 5th day
        ax.axvspan(9/24 + 4, 10/24 + 4, color='darkgrey', alpha=1, linewidth=0, zorder=10)
        ax.axvspan(9/24 + 5, 10/24 + 5, color='darkgrey', alpha=1, linewidth=0, zorder=11)

        # # add night injection timing on 4th and 5th day
        # ax.axvspan(23/24 + 3, 24/24 + 3, color='darkgrey', alpha=1, linewidth=0, zorder=10)
        # ax.axvspan(23/24 + 4, 24/24 + 4, color='darkgrey', alpha=1, linewidth=0, zorder=11)

        # add quantification timing on 4th and 5th day
        for epoch in measure_epochs:
            ax.axvspan(measure_epochs[epoch][0].hour/24 + measure_epochs[epoch][0].day -1,
                    measure_epochs[epoch][1].hour/24 + measure_epochs[epoch][0].day -1,
                    color='seagreen', alpha=0.5, linewidth=0, zorder=10)

        ax.set_ylim([0, 100])
        td = tv_internal.iloc[-1] - tv_internal.iloc[0]
        days = td.round('d')
        if td > days:
            days = days + '1d'
        days_to_plot = days.days + 1
        ax.set_xlim([1, days_to_plot - 16/24])
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
        plt.savefig(os.path.join(rootdir, "speed_30min_m-stdev_figure_condition_{0}.pdf".format(species_f.replace(' ', '-'))), dpi=350)
        plt.close()
    return


def weekly_individual_figure(rootdir, feature, fish_tracks_bin, change_times_m, bin_size_min=30):
    """ Plots the weekly data of each fish for each species

    :param rootdir:
    :param feature:
    :param fish_tracks_ds:
    :param species:
    :return:
    """

    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    num_day_bins, bins_per_h, _, _, _, _, _, _, border_bottom_week, border_top_week = peak_borders(bin_size_min, change_times_m)
    dawn_s, dawn_e, dusk_s, dusk_e = (change_times_m[0]-60)/60, (change_times_m[0]+60)/60, (change_times_m[3]-60)/60, (change_times_m[3] + 60) / 60

    date_form = DateFormatter("%H")

    for species_name in fish_tracks_bin.species.unique():
        fish_feature = fish_tracks_bin.loc[fish_tracks_bin.species == species_name, ['ts', 'FishID', feature]].pivot(
            columns='FishID', values=feature, index='ts')

        fig2, ax2 = plt.subplots(len(fish_feature.columns), 1, figsize=(2.5, 0.8*len(fish_feature.columns)))
        ax2[-1].set_ylabel(feature)
        for fish, fish_name in enumerate(fish_feature.columns):
            for day in range(7):
                ax2[fish].axvspan(dawn_s * bins_per_h + day * num_day_bins, dawn_e * bins_per_h + day * num_day_bins,
                            color='wheat', alpha=0.5, linewidth=0)
                ax2[fish].axvspan(dusk_s * bins_per_h + day * num_day_bins, dusk_e * bins_per_h + day * num_day_bins,
                                  color='wheat', alpha=0.5, linewidth=0)
                ax2[fish].axvspan(0 + day * num_day_bins, dawn_s * bins_per_h + day * num_day_bins,
                            color='lightblue', alpha=0.5, linewidth=0)
                ax2[fish].axvspan(dusk_e * bins_per_h + day * num_day_bins, num_day_bins + day * num_day_bins,
                            color='lightblue', alpha=0.5, linewidth=0)

            x = fish_feature.iloc[:, fish]
            ax2[fish].plot(np.arange(len(fish_feature.index)), x, linewidth=0.5)
            ax2[fish].title.set_text(fish_name)

            ax2[fish].xaxis.set_major_locator(MultipleLocator(num_day_bins))
            ax2[fish].xaxis.set_major_formatter(date_form)
            ax2[fish].set_xlim(0, num_day_bins*6)
            plt.xlabel("Time (h)")
            plt.ylabel("Speed (mm/s)")
            sns.despine(top=True, right=True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax2[fish].spines[axis].set_linewidth(0.5)
            ax2[fish].tick_params(width=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_weekly_individuals_{}_bin_size_{}min.pdf".format(species_name, bin_size_min)), dpi=350)
        plt.close()
    return
