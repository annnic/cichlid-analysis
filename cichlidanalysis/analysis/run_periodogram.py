import datetime as dt
import pandas as pd
import numpy as np
import csv

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.analysis.run_binned_als import load_bin_als_files, setup_run_binned
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.cosinorpy_periodogram import periodogram_df_an


def cosinor_sp(fish_tracks_bin, epoch, test_tag):
    all_species = fish_tracks_bin['species'].unique()

    first = True
    for species_f in all_species:

        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # need to use time divisible by 24h otherwise it doesn't look at those timepoints?
        filtered_spd = sp_spd[(epoch[0] < sp_spd.index) & (sp_spd.index < epoch[1])]

        # all days
        sp_spd_days = filtered_spd.mean(axis=1)
        sp_spd_ts = np.arange(0, len(sp_spd_days))/2

        # get time of day so that the same tod for each fish can be averaged
        filtered_spd.loc[:, 'time_of_day'] = filtered_spd.apply(lambda row: str(row.name)[11:16], axis=1)

        # # one day
        # sp_spd_ave = filtered_spd.groupby('time_of_day').mean()
        # sp_spd_ave_mean = sp_spd_ave.mean(axis=1)

        ##### cosinorpy periodogram analysis ########
        sp_spd_days_df = pd.DataFrame({'x': sp_spd_ts, 'y': sp_spd_days.reset_index(drop=True)})
        sp_spd_days_df['test'] = species_f

        peaks = periodogram_df_an(sp_spd_days_df, folder=rootdir, prefix=species_f, title=test_tag)

        if first:
            peaks_list = list(np.around(np.array(peaks), 2))
            peaks_list.sort()
            all_peaks = {species_f: peaks_list}
            first = False
        else:
            peaks_list = list(np.around(np.array(peaks), 2))
            peaks_list.sort()
            all_peaks[species_f] = peaks_list

    with open('periodogram_peaks_ordered.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(all_peaks.items())
    return


def cosinor_fish(fish_tracks_bin, epoch, test_tag, bin_size_min=30):
    all_species = fish_tracks_bin['species'].unique()
    bins_per_h = 60 / bin_size_min

    first_sp = True
    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # need to use time divisible by 24h otherwise it doesn't look at those timepoints?
        filtered_spd = sp_spd[(epoch[0] < sp_spd.index) & (sp_spd.index < epoch[1])]

        for fish, fish_name in enumerate(filtered_spd.columns):
            ##### cosinorpy periodogram analysis ########
            fish_spd = filtered_spd.loc[:, fish_name]
            sp_spd_ts = np.arange(0, len(fish_spd))/bins_per_h

            sp_spd_days_df = pd.DataFrame({'x': sp_spd_ts, 'y': fish_spd.reset_index(drop=True)})
            sp_spd_days_df['test'] = fish_name

            peaks = periodogram_df_an(sp_spd_days_df, folder=rootdir, prefix=fish_name, title=test_tag, save=False)

            if peaks[0]==False:
                peaks = [np.NaN, np.NaN]

            if first_sp:
                peaks_list = list(np.around(np.array(peaks), 2))
                peaks_list.sort()
                all_peaks = {fish_name: peaks_list}
                if 24.0 in peaks_list:
                    peak_24h = {'FishID': fish_name, '24h_peak': [1], 'species': species_f}
                    peaks_24h_df = pd.DataFrame.from_dict(peak_24h)
                else:
                    peak_24h = {'FishID': fish_name, '24h_peak': [0], 'species': species_f}
                    peaks_24h_df = pd.DataFrame.from_dict(peak_24h)
                first_sp = False
            else:
                peaks_list = list(np.around(np.array(peaks), 2))
                peaks_list.sort()
                all_peaks[fish_name] = peaks_list

                if 24.0 in peaks_list:
                    peak_24h = {'FishID': fish_name, '24h_peak': [1], 'species': species_f}
                    peaks_24h_df_fish = pd.DataFrame.from_dict(peak_24h)
                else:
                    peak_24h = {'FishID': fish_name, '24h_peak': [0], 'species': species_f}
                    peaks_24h_df_fish = pd.DataFrame.from_dict(peak_24h)
                peaks_24h_df = pd.concat([peaks_24h_df, peaks_24h_df_fish])

    with open('periodogram_peaks_individuals_ordered.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(all_peaks.items())
    return peaks_24h_df

if __name__ == '__main__':
    rootdir = select_dir_path()

    # fish_tracks_bin = load_bin_als_files(rootdir, "*als_30m.csv")
    # fish_tracks_bin_1m = load_bin_als_files(rootdir, "*als_1m.csv")

    # lose some species e.g. Astbur
    fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes = setup_run_binned(rootdir)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s, \
    change_times_d, change_times_m, change_times_datetime, change_times_unit = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # convert ts to datetime
    fish_tracks_bin['ts'] = pd.to_datetime(fish_tracks_bin['ts'])
    # fish_tracks_bin_1m['ts'] = pd.to_datetime(fish_tracks_bin_1m['ts'])

    # define day or nighttime
    fish_tracks_bin['time_of_day_m'] = fish_tracks_bin.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 +
                                                                            int(str(row)[11:16][-2:]))
    fish_tracks_bin['daynight'] = "d"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m > change_times_m[3], 'daynight'] = "n"
    print("added night and day column")

    # Light:dark
    epoch = [pd.to_datetime('1970-01-02 07:30:00'), pd.to_datetime('1970-01-05 8:00:00')]
    # dark:dark
    epoch = [pd.to_datetime('1970-01-02 07:30:00'), pd.to_datetime('1970-01-05 8:00:00')]
    # full for standard
    epoch = [pd.to_datetime('1970-01-02 07:30:00'), pd.to_datetime('1970-01-05 8:00:00')]
    tag1 = 'light-dark'
    tag2 = 'dark-dark'
    tag3 = 'six_days'
    tag4 = 'five_days'

    # CHECK!!!! May take the one after to
    epochs = {tag1: [pd.to_datetime('1970-01-02 07:30:00'), pd.to_datetime('1970-01-05 8:00:00')],
              tag2: [pd.to_datetime('1970-01-05 07:30:00'), pd.to_datetime('1970-01-08 8:00:00')],
              tag3: [pd.to_datetime('1970-01-02 00:00:00'), pd.to_datetime('1970-01-08 00:30:00')],
              tag4: [pd.to_datetime('1970-01-02 00:00:00'), pd.to_datetime('1970-01-07 00:30:00')]}

    ##### CHANGE HERE!
    test_tag = tag3

    epoch = epochs[test_tag]

    #### cosinor analysis
    # cosinor_sp(fish_tracks_bin, epoch, test_tag)

    peaks_24h_df = cosinor_fish(fish_tracks_bin, epoch, test_tag)
    print('yay')
    peaks_24h_df = peaks_24h_df.reset_index(drop=True)

    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.figure(figsize=(3, 6))

    # total = peaks_24h_df.groupby('species')['24h_peak'].sum().reset_index()
    total = peaks_24h_df.groupby('species').size().reset_index(name='total_counts')
    circadian_peak = peaks_24h_df[peaks_24h_df.loc[:, '24h_peak'] == 1].groupby('species').size().reset_index(name='circ_counts')
    circadian_peak_df = pd.merge(total, circadian_peak, how='left')

    bar1 = sns.barplot(x="total_counts", y="species", data=circadian_peak_df, color='darkblue')
    bar2 = sns.barplot(x="circ_counts", y="species", data=circadian_peak_df, estimator=sum, ci=None, color='lightblue')

    top_bar = mpatches.Patch(color='darkblue', label='24h peak = No')
    bottom_bar = mpatches.Patch(color='lightblue', label='24h peak = Yes')
    plt.legend(handles=[top_bar, bottom_bar])

    for axis in ['top', 'bottom', 'left', 'right']:
        bar1.spines[axis].set_linewidth(0.5)
    bar1.tick_params(width=0.5)
    bar1.spines['top'].set_visible(False)
    bar1.spines['right'].set_visible(False)

    plt.savefig(os.path.join(rootdir, "individual_periodogram_24h_peaks.pdf"), dpi=350)
    plt.close()



