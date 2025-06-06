import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import matplotlib
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches


def daily_ave_spd(sp_spd_ave, sp_spd_ave_std, rootdir, species_f, change_times_unit):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    daily_speed = sp_spd_ave.mean(axis=1)

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_spd_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_spd_ave.index, y=sp_spd_ave.iloc[:, cols], color='tab:blue', alpha=0.3)
    ax = sns.lineplot(x=sp_spd_ave.index, y=daily_speed, linewidth=4, color='tab:blue')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Speed (mm/s)")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_individual_{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed + sp_spd_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed - sp_spd_ave_std), color='lightgrey')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Speed (mm/s)")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()


def daily_ave_move(sp_move_ave, sp_move_ave_std, rootdir, species_f, change_times_unit):
    """

    :param sp_move_ave:
    :param sp_move_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return:
    """
    daily_move = sp_move_ave.mean(axis=1)

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_move_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_move_ave.index, y=(sp_move_ave).iloc[:, cols], color='palevioletred', alpha=0.3)
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move), color='palevioletred')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Movement")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "movement_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # movement (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move + sp_move_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move - sp_move_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_move_ave.index, y=(daily_move), color='palevioletred')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Movement")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "movement_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()


def daily_ave_rest(sp_rest_ave, sp_rest_ave_std, rootdir, species_f, change_times_unit):
    daily_rest = sp_rest_ave.mean(axis=1)

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_rest_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_rest_ave.index, y=(sp_rest_ave).iloc[:, cols], color='darkorchid', alpha=0.3)
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest), color='darkorchid')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Rest")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "Rest_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # rest (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest + sp_rest_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest - sp_rest_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_rest_ave.index, y=(daily_rest), color='darkorchid')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Rest")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "rest_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()
    return daily_rest


def daily_ave_vp(rootdir, sp_vp_ave, sp_vp_ave_std, species_f, change_times_unit):
    daily_rest = sp_vp_ave.mean(axis=1)

    plt.figure(figsize=(6, 4))
    for cols in np.arange(0, sp_vp_ave.columns.shape[0]):
        ax = sns.lineplot(x=sp_vp_ave.index, y=(sp_vp_ave).iloc[:, cols], color='teal', alpha=0.3)
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest), color='teal')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Vertical position")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "vp_30min_ave_individual{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()

    # rest (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest + sp_vp_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest - sp_vp_ave_std), color='lightgrey')
    ax = sns.lineplot(x=sp_vp_ave.index, y=(daily_rest), color='teal')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h:m)")
    plt.ylabel("Vertical position")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    plt.title(species_f)
    plt.savefig(os.path.join(rootdir, "vp_30min_ave_ave-stdev{0}.png".format(species_f.replace(' ', '-'))))
    plt.close()
    return daily_rest


def plot_daily(fish_tracks_30m_i, change_times_unit, rootdir):

    all_species = fish_tracks_30m_i['species'].unique()

    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_spd['time_of_day'] = sp_spd.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_ave = sp_spd.groupby('time_of_day').mean()
        sp_spd_ave_std = sp_spd_ave.std(axis=1)

        # make the plots
        daily_ave_spd(sp_spd_ave, sp_spd_ave_std, rootdir, species_f, change_times_unit)
        daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit)
        daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=100)
        daily_ave_spd_figure_sex(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=100)

        # ### movement ###
        move = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['movement', 'FishID', 'ts']]
        sp_move = move.pivot(columns='FishID', values='movement', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_move['time_of_day'] = sp_move.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_move_ave = sp_move.groupby('time_of_day').mean()
        sp_move_ave_std = sp_move_ave.std(axis=1)

        # make the plots
        daily_ave_move(sp_move_ave, sp_move_ave_std, rootdir, species_f, change_times_unit)

        # ### rest ###
        rest = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['rest', 'FishID', 'ts']]
        sp_rest = rest.pivot(columns='FishID', values='rest', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_rest['time_of_day'] = sp_rest.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_rest_ave = sp_rest.groupby('time_of_day').mean()
        sp_rest_ave_std = sp_rest_ave.std(axis=1)

        # make the plots
        daily_ave_rest(sp_rest_ave, sp_rest_ave_std, rootdir, species_f, change_times_unit)

        # ### vertical position ###
        vertical_pos = fish_tracks_30m_i[fish_tracks_30m_i.species == species_f][['vertical_pos', 'FishID', 'ts']]
        sp_vp = vertical_pos.pivot(columns='FishID', values='vertical_pos', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_vp['time_of_day'] = sp_vp.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_vp_ave = sp_vp.groupby('time_of_day').mean()
        sp_vp_ave_std = sp_vp_ave.std(axis=1)

        # make the plots
        daily_ave_vp(rootdir, sp_vp_ave, sp_vp_ave_std, species_f, change_times_unit)
    return


def daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=60):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    date_form = DateFormatter("%H")

    daily_speed = sp_spd_ave.mean(axis=1)

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(1.2, 1.2))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed + sp_spd_ave_std), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed - sp_spd_ave_std), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed), linewidth=0.5)

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, ymax])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h)", fontsize=SMALL_SIZE)
    plt.ylabel("Speed (mm/s)", fontsize=SMALL_SIZE)
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

    ax.xaxis.set_major_locator(MultipleLocator(24))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev_figure_{0}_{1}.pdf".format(species_f.replace(' ', '-'),
                                                                                            ymax)))
    plt.close()
    return


def daily_ave_spd_figure_sex(rootdir, sp_spd_daily, sp_spd_daily_std, species_f, change_times_unit, fish_num,
                             diel_guilds, cichlid_meta, temporal_col, ymax=60):
    """ speed_mm (30m bins daily average) for each fish (individual lines) with sex split and coloured by diel guild
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    date_form = DateFormatter("%H")

    colours = {'male': 'cornflowerblue', 'female': 'mediumorchid', 'unknown': 'gold', 'm': 'cornflowerblue',
               'f': 'mediumorchid', 'u': 'gold'}

    temporal_colors = diel_guilds.diel_guild.map(temporal_col)
    temporal_colors = temporal_colors.set_axis(diel_guilds.species)

    rect_x = 0  # x-coordinate of the bottom-left corner of the rectangle
    rect_y = ymax  # y-coordinate of the bottom-left corner of the rectangle
    rect_width = 48  # width of the rectangle
    rect_height = 2  # height of the rectangle

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(1.2, 1.2))
    for sex in sp_spd_daily.columns:
        ax = sns.lineplot(x=sp_spd_daily.index, y=(sp_spd_daily.loc[:, sex] + sp_spd_daily_std.loc[:, sex]), color=colours[sex], linewidth=0.5, alpha=0.35)
        ax = sns.lineplot(x=sp_spd_daily.index, y=(sp_spd_daily.loc[:, sex] - sp_spd_daily_std.loc[:, sex]), color=colours[sex], linewidth=0.5, alpha=0.35)
        ax = sns.lineplot(x=sp_spd_daily.index, y=(sp_spd_daily.loc[:, sex]),  color=colours[sex], linewidth=0.75)

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_ylim([0, ymax])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h)", fontsize=SMALLEST_SIZE)
    plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)

    # get all names
    sp_meta = cichlid_meta.loc[cichlid_meta.species_our_names == species_f, :]
    plt.title(sp_meta.species_true.values[0] + '\n' + ' (' + sp_meta.six_letter_name_Ronco.values[0] + ')', fontsize=SMALLEST_SIZE)

    # add N for each
    # add N number
    for i, sex in enumerate(('m', 'f', 'u')):
        ax.text(1, ymax-12-i*10, sex + ': ' + str(fish_num[sex]), fontsize=SMALLEST_SIZE, color=colours[sex])

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

    ax.xaxis.set_major_locator(MultipleLocator(24))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev_sex_figure_{0}_{1}.pdf".format(species_f.replace(' ', '-'),
                                                                                            ymax)))
    plt.close()
    return


def daily_ave_spd_figure_night_centred(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=60):
    """ speed_mm (30m bins daily average) for each fish (individual lines), but now night centered

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    date_form = DateFormatter("%H")

    # move the first rows to the end to centre midnight
    sp_spd_ave_midnight = pd.concat([sp_spd_ave.iloc[24:], sp_spd_ave.iloc[:24]])
    sp_spd_ave_std_midnight = pd.concat([sp_spd_ave_std.iloc[24:], sp_spd_ave_std.iloc[:24]])
    daily_speed_midnight = sp_spd_ave_midnight.mean(axis=1)

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(1.2, 1.2))
    ax = sns.lineplot(x=sp_spd_ave_midnight.index, y=(daily_speed_midnight + sp_spd_ave_std_midnight), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave_midnight.index, y=(daily_speed_midnight - sp_spd_ave_std_midnight), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave_midnight.index, y=(daily_speed_midnight), linewidth=0.5)

    # changed the plotting
    # ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[1], change_times_unit[2], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    # ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)
    # ax.set_ylim([0, ymax])
    ax.set_xlim([0, 24 * 2])
    plt.xlabel("Time (h)", fontsize=SMALL_SIZE)
    plt.ylabel("Speed (mm/s)", fontsize=SMALL_SIZE)
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

    ax.xaxis.set_major_locator(MultipleLocator(24))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev_figure_{0}_{1}_midnight_ylim_adapt.pdf".format(species_f.replace(' ', '-'),
                                                                                            ymax)))
    plt.close()
    return

def daily_ave_spd_figure_timed_perturb(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=60,
                                       label='_', perturb_timing=[0, 0]):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    date_form = DateFormatter("%H")

    daily_speed = sp_spd_ave.mean(axis=1)

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(1, 1))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed + sp_spd_ave_std), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed - sp_spd_ave_std), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed), linewidth=0.5)

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)

    ax.axvspan(perturb_timing[0], perturb_timing[1], color='m', alpha=0.5, linewidth=0)

    ax.set_ylim([0, ymax])
    ax.set_xlim([0, 24 * 2])
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

    ax.xaxis.set_major_locator(MultipleLocator(24))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev_figure_light_perturb_{0}_{1}{2}.pdf".format(species_f.replace(' ', '-'),
                                                                                            ymax, label)))
    plt.close()
    return


def daily_ave_spd_figure_timed_double(rootdir, sp_spd_ave_tag1, sp_spd_ave_std_tag1, sp_spd_ave_tag2,
                                      sp_spd_ave_std_tag2, species_f, change_times_unit, ymax=60, label='_',
                                      perturb_timing=[0, 0]):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    date_form = DateFormatter("%H")

    daily_speed_tag1 = sp_spd_ave_tag1.mean(axis=1)
    daily_speed_tag2 = sp_spd_ave_tag2.mean(axis=1)

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(1, 1))
    ax = sns.lineplot(x=sp_spd_ave_tag1.index, y=(daily_speed_tag1 + sp_spd_ave_std_tag1), color='m', alpha=0.2,
                      linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave_tag1.index, y=(daily_speed_tag1 - sp_spd_ave_std_tag1), color='m', alpha=0.2,
                      linewidth=0.5)

    ax = sns.lineplot(x=sp_spd_ave_tag2.index, y=(daily_speed_tag2 + sp_spd_ave_std_tag2), color='k', alpha=0.2,
                      linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave_tag2.index, y=(daily_speed_tag2 - sp_spd_ave_std_tag2), color='k', alpha=0.2,
                      linewidth=0.5)

    ax = sns.lineplot(x=sp_spd_ave_tag1.index, y=(daily_speed_tag1), linewidth=0.5, color='m')
    ax = sns.lineplot(x=sp_spd_ave_tag2.index, y=(daily_speed_tag2), linewidth=0.5, color='k')

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)

    ax.axvspan(perturb_timing[0], perturb_timing[1], color='darkgrey', alpha=1, linewidth=0, zorder=10)

    ax.set_ylim([0, ymax])
    ax.set_xlim([0, 24 * 2])
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

    ax.xaxis.set_major_locator(MultipleLocator(24))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev_figure_condition_{0}_{1}{2}.pdf".format(
        species_f.replace(' ', '-'), ymax, label)))
    plt.close()
    return

def daily_ave_spd_figure_timed_perturb_darkdark(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit,
                                                ymax=60, label='_', perturb_timing=[0, 0]):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param sp_spd_ave:
    :param sp_spd_ave_std:
    :param rootdir:
    :param species_f:
    :param change_times_unit:
    :return: daily_speed:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    date_form = DateFormatter("%H")
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    daily_speed = sp_spd_ave.mean(axis=1)

    # speed_mm (30m bins daily average) for each fish (mean  +- std)
    plt.figure(figsize=(1, 1))
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed + sp_spd_ave_std), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed - sp_spd_ave_std), color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=sp_spd_ave.index, y=(daily_speed), linewidth=0.5)

    ax.axvspan(0, change_times_unit[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_unit[0], change_times_unit[3], color='lightblue', alpha=0.25, linewidth=0)
    ax.axvspan(change_times_unit[3], 24 * 2, color='lightblue', alpha=0.5, linewidth=0)

    ax.axvspan(perturb_timing[0], perturb_timing[1], color='m', alpha=0.5, linewidth=0)

    ax.set_ylim([0, ymax])
    ax.set_xlim([0, 24 * 2])
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

    ax.xaxis.set_major_locator(MultipleLocator(24))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "speed_30min_ave_ave-stdev_figure_light_darkdark_{0}_{1}{2}.pdf".format(
        species_f.replace(' ', '-'), ymax, label)))
    plt.close()
    return
