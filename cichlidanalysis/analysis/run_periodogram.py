import datetime as dt
import pandas as pd
import numpy as np

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.analysis.run_binned_als import load_bin_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.cosinorpy_periodogram import periodogram_df_an


if __name__ == '__main__':
    rootdir = select_dir_path()

    fish_tracks_bin = load_bin_als_files(rootdir, "*als_30m.csv")
    # fish_tracks_bin_1m = load_bin_als_files(rootdir, "*als_1m.csv")

    fish_IDs = fish_tracks_bin['FishID'].unique()

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


    epoch = [pd.to_datetime('1970-01-02 07:30:00'), pd.to_datetime('1970-01-05 8:00:00')]

    #### cosinor analysis
    all_species = fish_tracks_bin['species'].unique()

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

        periodogram_df_an(sp_spd_days_df, folder=rootdir, prefix=species_f)
