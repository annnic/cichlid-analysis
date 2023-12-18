import os

import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind
from scipy import stats


from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.analysis.run_binned_als import load_bin_als_files
from cichlidanalysis.utils.timings import load_timings_14_8
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_mstd_figure, plot_speed_30m_mstd_figure_light_perturb
from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure_timed_perturb, daily_ave_spd_figure_timed_perturb_darkdark
from cichlidanalysis.plotting.plot_light_perturb import plot_ld_dd_stripplot, plot_ld_dd_dn_dif_stripplot


if __name__ == '__main__':
    rootdir = select_dir_path()

    fish_tracks_bin = load_bin_als_files(rootdir, "*als_30m.csv")

    fish_IDs = fish_tracks_bin['FishID'].unique()
    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s, \
    change_times_d, change_times_m, change_times_datetime, change_times_unit = load_timings_14_8(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # convert ts to datetime
    fish_tracks_bin['ts'] = pd.to_datetime(fish_tracks_bin['ts'])

    ### need to convert tv from str to datetime
    # speed_mm (30m bins) for each species (mean  +- std)
    plot_speed_30m_mstd_figure_light_perturb(rootdir, fish_tracks_bin, change_times_d)
    # plot_speed_30m_mstd_figure(rootdir, fish_tracks_bin, change_times_d)

    # day 2 to 8am on day 5 = baseline
    # day 4 8am until end
    epochs = {'epoch_1': [pd.to_datetime('1970-01-02 00:00:00'), pd.to_datetime('1970-01-05 08:00:00')],
              'epoch_2': [pd.to_datetime('1970-01-05 07:30:00'), pd.to_datetime('1970-01-07 08:00:00')]}

    all_species = fish_tracks_bin['species'].unique()

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


    tag1 = 'control'
    tag2 = 'dark:dark'
    epochs = {tag1: [pd.to_datetime('1970-01-02 00:00:00'), pd.to_datetime('1970-01-05 8:00:00')],
              tag2: [pd.to_datetime('1970-01-05 8:00:00'), pd.to_datetime('1970-01-08 8:00:00')]}

    # define day or nighttime
    fish_tracks_bin['time_of_day_m'] = fish_tracks_bin.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 +
                                                                            int(str(row)[11:16][-2:]))
    fish_tracks_bin['daynight'] = "d"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m > change_times_m[3], 'daynight'] = "n"
    print("added night and day column")

    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})
    custom_palette = ["gold", "grey"]
    custom_order = [tag1, tag2]

    # For each species, for each individual, for each epoch, find the difference between day and night speed
    for species_f in all_species:
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts', 'daynight']]

        spd_d = spd[spd.daynight == 'd']
        spd_n = spd[spd.daynight == 'n']

        sp_spd_d = spd_d.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_n = spd_n.pivot(columns='FishID', values='speed_mm', index='ts')

        for epoch_n, epoch in enumerate(epochs):

            epoch_spd_d = sp_spd_d[(epochs[epoch][0] < sp_spd_d.index) & (sp_spd_d.index < epochs[epoch][1])]
            epoch_spd_n = sp_spd_n[(epochs[epoch][0] < sp_spd_n.index) & (sp_spd_n.index < epochs[epoch][1])]

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
        custom_palette_2 = ["gold", "lightblue", "#EAF5F9", "lightblue"]

        plot_ld_dd_stripplot(rootdir, spd_aves_dn, custom_palette_2, SMALLEST_SIZE, species_f, ttest_control, ttest_dd)

        #### cosinor analysis
        from CosinorPy import cosinor, cosinor1
        import numpy as np

        for species_f in all_species:
            # ### speed ###
            spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts']]
            sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

            for epoch in epochs:
                filtered_spd = sp_spd[(epochs[epoch][0] < sp_spd.index) & (sp_spd.index < epochs[epoch][1])]

                # get time of day so that the same tod for each fish can be averaged
                filtered_spd.loc[:, 'time_of_day'] = filtered_spd.apply(lambda row: str(row.name)[11:16], axis=1)
                sp_spd_ave = filtered_spd.groupby('time_of_day').mean()
                sp_spd_ave_mean = sp_spd_ave.mean(axis=1)

        cosinor1.fit_cosinor(np.arange(0, 48), sp_spd_ave_mean.reset_index(drop=True), 48, save_to=(rootdir + "/test"), plot_on=True)

        plt.savefig(os.path.join(rootdir, "test_{0}.pdf".format(species_f.replace(' ', '-'))),dpi=350)
