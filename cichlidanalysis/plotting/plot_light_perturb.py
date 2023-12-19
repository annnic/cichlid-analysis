import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy import stats

from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure_timed_perturb, daily_ave_spd_figure_timed_perturb_darkdark


def plot_ld_dd_stripplot(rootdir, spd_aves_dn, custom_palette_2, SMALLEST_SIZE, species_f, ttest_control, ttest_dd):
    plt.figure(figsize=(1.5, 1))
    ax = sns.stripplot(data=spd_aves_dn, x='condition', y='speed_mm', hue='condition', s=2,
                       palette=custom_palette_2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0, 80])
    plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)
    plt.title(species_f, fontsize=SMALLEST_SIZE)

    # Decrease the offset for tick labels on all axes
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5

    # Adjust the offset for tick labels on all axes
    ax.tick_params(axis='x', pad=0.5, length=2)
    ax.tick_params(axis='y', pad=0.5, length=2)

    plt.axhline(y=0, color='silver', linestyle='-', linewidth=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(-0.3, 80, round(ttest_control[1], 3))
    plt.text(1.7, 80, round(ttest_dd[1], 3))
    plt.tight_layout()
    plt.savefig(
        os.path.join(rootdir, "speed_figure_ld-dd_dn_stripplot_10-14h_{0}.pdf".format(species_f.replace(' ', '-'))),
        dpi=350)
    plt.close()
    return


def plot_ld_dd_dn_dif_stripplot(rootdir, spd_aves_condition, custom_palette, custom_order, SMALLEST_SIZE, species_f):
    plt.figure(figsize=(1, 1))
    ax = sns.stripplot(data=spd_aves_condition, x='epoch', y='spd_ave_dif', s=2,
                       palette=custom_palette,
                       order=custom_order)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([-50, 28])
    plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)
    plt.title(species_f, fontsize=SMALLEST_SIZE)

    # Decrease the offset for tick labels on all axes
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5

    # Adjust the offset for tick labels on all axes
    ax.tick_params(axis='x', pad=0.5, length=2)
    ax.tick_params(axis='y', pad=0.5, length=2)

    plt.axhline(y=0, color='silver', linestyle='-', linewidth=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(
        os.path.join(rootdir,
                     "speed_figure_ld-dd_dif_stripplot_10-14h_{0}.pdf".format(species_f.replace(' ', '-'))),
        dpi=350)
    plt.close()
    return


def plot_daily_activity_light(rootdir, fish_tracks_bin, epochs, change_times_unit):
    all_species = fish_tracks_bin['species'].unique()
    # plot daily activity patterns for each epoch
    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        for epoch in epochs:
            filtered_spd = sp_spd[(epochs[epoch][0] < sp_spd.index) & (sp_spd.index < epochs[epoch][1])]

            # get time of day so that the same tod for each fish can be averaged
            filtered_spd.loc[:, 'time_of_day'] = filtered_spd.apply(lambda row: str(row.name)[11:16], axis=1)
            sp_spd_ave = filtered_spd.groupby('time_of_day').mean()
            sp_spd_ave_std = sp_spd_ave.std(axis=1)

            daily_ave_spd_figure_timed_perturb(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit,
                                               ymax=60, label=epoch)
            daily_ave_spd_figure_timed_perturb(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit,
                                               ymax=100, label=epoch)

            daily_ave_spd_figure_timed_perturb_darkdark(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit,
                                               ymax=100, label=epoch)
    return


def plot_stripplots_light_perturb(rootdir, fish_tracks_bin, tag1, tag2, epochs_color):
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})
    custom_palette = ["gold", "grey"]
    custom_order = [tag1, tag2]

    all_species = fish_tracks_bin['species'].unique()
    # For each species, for each individual, for each epoch, find the difference between day and night speed
    for species_f in all_species:
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts', 'daynight']]

        spd_d = spd[spd.daynight == 'd']
        spd_n = spd[spd.daynight == 'n']

        sp_spd_d = spd_d.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_n = spd_n.pivot(columns='FishID', values='speed_mm', index='ts')

        for epoch_n, epoch in enumerate(epochs_color):

            epoch_spd_d = sp_spd_d[(epochs_color[epoch][0] < sp_spd_d.index) & (sp_spd_d.index < epochs_color[epoch][1])]
            epoch_spd_n = sp_spd_n[(epochs_color[epoch][0] < sp_spd_n.index) & (sp_spd_n.index < epochs_color[epoch][1])]

            spd_ave_d = epoch_spd_d.mean(axis=0)
            spd_ave_n = epoch_spd_n.mean(axis=0)

            spd_ave_dif = spd_ave_d - spd_ave_n

            spd_ave_d = pd.DataFrame(spd_ave_d, columns=['speed_mm'])
            spd_ave_d['epoch'] = epoch
            spd_ave_d['daytime'] = 'd'
            spd_ave_d['condition'] = 'd_{}'.format(epoch)

            spd_ave_n = pd.DataFrame(spd_ave_n, columns=['speed_mm'])
            spd_ave_n['epoch'] = epoch
            spd_ave_n['daytime'] = 'n'
            spd_ave_n['condition'] = 'n_{}'.format(epoch)
            spd_dn = pd.concat([spd_ave_d, spd_ave_n])

            df_dif = pd.DataFrame({'spd_ave_dif': spd_ave_dif, 'epoch': epoch})

            if epoch_n == 0:
                spd_aves_condition = df_dif
                spd_aves_dn = spd_dn
            else:
                spd_aves_condition = pd.concat([spd_aves_condition, df_dif])
                spd_aves_dn = pd.concat([spd_aves_dn, spd_dn])

        # calculate dif between day and night for each day
        ############## plot difference
        plot_ld_dd_dn_dif_stripplot(rootdir, spd_aves_condition, custom_palette, custom_order, SMALLEST_SIZE, species_f)

        ############## plot day and night
        ttest_control = stats.ttest_rel(spd_aves_dn.speed_mm[spd_aves_dn.condition == 'd_control'],
                                        spd_aves_dn.speed_mm[spd_aves_dn.condition == 'n_control'])
        ttest_dd = stats.ttest_rel(spd_aves_dn.speed_mm[spd_aves_dn.condition == 'd_dark:dark'],
                                   spd_aves_dn.speed_mm[spd_aves_dn.condition == 'n_dark:dark'])
        custom_palette_2 = ["gold", "lightblue", "cadetblue", "lightblue"]  # "#EAF5F9",

        plot_ld_dd_stripplot(rootdir, spd_aves_dn, custom_palette_2, SMALLEST_SIZE, species_f, ttest_control, ttest_dd)
    return
