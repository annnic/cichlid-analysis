import os

import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.als_files import load_bin_als_files, load_bin_als_file_condition
from cichlidanalysis.utils.timings import load_timings_14_8, load_timings
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_mstd_figure_light_perturb, \
    plot_speed_30m_mstd_figure_conditions
from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure_timed_perturb, daily_ave_spd_figure_timed_double


if __name__ == '__main__':
    rootdir = select_dir_path()

    tag1 = 'melatonin'
    tag2 = 'ethanol'
    # fish_tracks_bin = load_bin_als_files(rootdir, "*als_30m.csv")
    fish_tracks_bin = load_bin_als_file_condition(rootdir, suffix="*als_30m.csv", tag1=tag1, tag2=tag2)

    fish_IDs = fish_tracks_bin['FishID'].unique()
    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s, \
    change_times_d, change_times_m, change_times_datetime, change_times_unit = load_timings_14_8(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # convert ts to datetime
    fish_tracks_bin['ts'] = pd.to_datetime(fish_tracks_bin['ts'])

    # day
    measure_epochs = {'epoch_1': [pd.to_datetime('1970-01-05 12:00:00'), pd.to_datetime('1970-01-05 16:00:00')],
              'epoch_2': [pd.to_datetime('1970-01-06 12:00:00'), pd.to_datetime('1970-01-06 16:00:00')]}

    # # day 12-12h LD
    # measure_epochs = {'epoch_1': [pd.to_datetime('1970-01-05 10:00:00'), pd.to_datetime('1970-01-05 14:00:00')],
    #           'epoch_2': [pd.to_datetime('1970-01-06 10:00:00'), pd.to_datetime('1970-01-06 14:00:00')]}

    # # #night
    # measure_epochs = {'epoch_1': [pd.to_datetime('1970-01-05 00:00:00'), pd.to_datetime('1970-01-05 04:00:00')],
    #           'epoch_2': [pd.to_datetime('1970-01-06 00:00:00'), pd.to_datetime('1970-01-06 04:00:00')]}

    # speed_mm (30m bins) for each species (mean  +- std)
    plot_speed_30m_mstd_figure_conditions(rootdir, fish_tracks_bin, change_times_d, tag1, tag2, measure_epochs)

    # day 2 to 8am on day 5 = baseline
    # day 4 8am until end
    epochs = {'epoch_1': [pd.to_datetime('1970-01-02 00:00:00'), pd.to_datetime('1970-01-05 08:00:00')],
              'epoch_2': [pd.to_datetime('1970-01-05 07:30:00'), pd.to_datetime('1970-01-07 08:00:00')]}

    all_species = fish_tracks_bin['species'].unique()

    # daily plots individual conditions, both epochs
    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        for epoch_n, epoch in enumerate(epochs):
            filtered_spd = sp_spd[(epochs[epoch][0] < sp_spd.index) & (sp_spd.index < epochs[epoch][1])]

            # get time of day so that the same tod for each fish can be averaged
            filtered_spd['time_of_day'] = filtered_spd.apply(lambda row: str(row.name)[11:16], axis=1)
            sp_spd_ave = filtered_spd.groupby('time_of_day').mean()
            sp_spd_ave_std = sp_spd_ave.std(axis=1)

            if epoch_n == 0:
                perturb_timing = [0, 0]
            elif epoch_n == 1:
                perturb_timing = [18, 19]

            daily_ave_spd_figure_timed_perturb(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit,
                                               ymax=60, label=epoch, perturb_timing=perturb_timing)
            daily_ave_spd_figure_timed_perturb(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit,
                                               ymax=100, label=epoch, perturb_timing=perturb_timing)

    # daily plots combined conditions, both epochs
    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts', 'condition']]
        spd_tag1 = spd.loc[spd.condition == tag1]
        spd_tag2 = spd.loc[spd.condition == tag2]
        sp_spd_tag1 = spd_tag1.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_tag2 = spd_tag2.pivot(columns='FishID', values='speed_mm', index='ts')

        for epoch_n, epoch in enumerate(epochs):
            filtered_spd_tag1 = sp_spd_tag1[(epochs[epoch][0] < sp_spd_tag1.index) & (sp_spd_tag1.index < epochs[epoch][1])]
            filtered_spd_tag2 = sp_spd_tag2[(epochs[epoch][0] < sp_spd_tag2.index) & (sp_spd_tag2.index < epochs[epoch][1])]

            # get time of day so that the same tod for each fish can be averaged
            filtered_spd_tag1['time_of_day'] = filtered_spd_tag1.apply(lambda row: str(row.name)[11:16], axis=1)
            sp_spd_ave_tag1 = filtered_spd_tag1.groupby('time_of_day').mean()
            sp_spd_ave_std_tag1 = sp_spd_ave_tag1.std(axis=1)

            filtered_spd_tag2['time_of_day'] = filtered_spd_tag2.apply(lambda row: str(row.name)[11:16], axis=1)
            sp_spd_ave_tag2 = filtered_spd_tag2.groupby('time_of_day').mean()
            sp_spd_ave_std_tag2 = sp_spd_ave_tag2.std(axis=1)

            if epoch_n == 0:
                perturb_timing = [0, 0]
            elif epoch_n == 1:
                perturb_timing = [18, 19]

            daily_ave_spd_figure_timed_double(rootdir, sp_spd_ave_tag1, sp_spd_ave_std_tag1, sp_spd_ave_tag2,
                                              sp_spd_ave_std_tag2, species_f, change_times_unit, ymax=70, label=epoch,
                                              perturb_timing=perturb_timing)
            daily_ave_spd_figure_timed_double(rootdir, sp_spd_ave_tag1, sp_spd_ave_std_tag1, sp_spd_ave_tag2,
                                              sp_spd_ave_std_tag2, species_f, change_times_unit, ymax=70, label=epoch,
                                              perturb_timing=perturb_timing)


    # grab ethonal and melatonin at 10-14 and plot and compare, from day 4 and day 5
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})
    custom_palette = ["k", "m", "darkgrey", "violet"]
    custom_order = [tag2, tag1]
    ymax = 70

    # fish_tracks_bin, change_times_d, tag1, tag2
    for species_f in all_species:
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts', 'condition']]
        spd_tag1 = spd.loc[spd.condition == tag1]
        spd_tag2 = spd.loc[spd.condition == tag2]
        sp_spd_tag1 = spd_tag1.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_tag2 = spd_tag2.pivot(columns='FishID', values='speed_mm', index='ts')

        for epoch_n, epoch in enumerate(measure_epochs):
                filtered_spd_tag1 = sp_spd_tag1[(measure_epochs[epoch][0] < sp_spd_tag1.index) & (sp_spd_tag1.index < measure_epochs[epoch][1])]
                spd_tag1_ave = filtered_spd_tag1.mean(axis=0)

                filtered_spd_tag2 = sp_spd_tag2[(measure_epochs[epoch][0] < sp_spd_tag2.index) & (sp_spd_tag2.index < measure_epochs[epoch][1])]
                spd_tag2_ave = filtered_spd_tag2.mean(axis=0)

                df_1 = pd.DataFrame({'spd_ave': spd_tag1_ave, 'condition': tag1, 'epoch': epoch, 'col_tag': tag1 + '-'
                                                                                                            + epoch})
                df_2 = pd.DataFrame({'spd_ave': spd_tag2_ave, 'condition': tag2, 'epoch': epoch, 'col_tag': tag2 + '-'
                                                                                                            + epoch})
                if epoch_n == 0:
                    spd_aves_condition = pd.concat([df_2, df_1])
                else:
                    spd_aves_condition_i = pd.concat([df_2, df_1])
                    spd_aves_condition = pd.concat([spd_aves_condition, spd_aves_condition_i])

                # t_statistic, p_value = ttest_ind(group1, group2)

        # 2 epochs, 2 conditions, 2 species - sperarate species
        ttest_stats = ttest_ind(spd_aves_condition.spd_ave[spd_aves_condition.condition == tag2],
                        spd_aves_condition.spd_ave[spd_aves_condition.condition == tag1])

        plt.figure(figsize=(1.8, 1))
        ax = sns.stripplot(data=spd_aves_condition, x='condition', y='spd_ave', s=2, hue='col_tag',
                           palette=custom_palette,
                           order=custom_order)
        plt.title("")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim([0, ymax])
        plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)
        # plt.title(species_f, fontsize=SMALLEST_SIZE)
        # ax.set_xticklabels(custom_order, rotation=45)

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

        plt.text(-0.2, ymax, 'P-value: ' + str(round(ttest_stats[1], 3)))

        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "speed_figure_condition_stripplot_10-14h_{0}.pdf".format(species_f.replace(' ', '-'))), dpi=350)
        plt.close()

