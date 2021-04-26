import os

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.gridspec as grid_spec
import datetime as dt
from datetime import timedelta

from cichlidanalysis.plotting.single_plots import fill_plot_ts
from cichlidanalysis.utils.species_names import shorten_sp_name


# speed_mm (30m bins) for each fish (individual lines)
def plot_speed_30m_individuals(rootdir, fish_tracks_30m, change_times_d):
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


def plot_spd_30min_combined(fish_tracks_ds_i, feature, ymax, span_max, ylabeling, change_times_datetime_i, rootdir):
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
        averages[species_n, :] = average[0:303]
        stdv = sp_feature.std(axis=1)

        # create time vector in datetime format
        # tv = fish_tracks_ds.loc[fish_tracks_ds.FishID == fish_IDs[0], 'ts']
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

        short_name = shorten_sp_name(species_name)
        shortened_sp_name = species_name[0] + ". " + species_name.split(' ')[1]
        ax_objs[-1].text(0.9, 0, short_name[0], fontweight="bold", fontsize=10, ha="right", rotation=-45)
        gs.update(hspace=-0.1)
    plt.show()

    plt.savefig(os.path.join(rootdir, "speed_30min_combined_species_{0}.png".format(dt.date.today())))
    plt.close()
    aves_feature = pd.DataFrame(averages.T, columns=species, index=date_time_obj[0:averages.shape[1]])
    return aves_feature, date_time_obj, sp_feature_combined
