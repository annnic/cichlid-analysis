import warnings
import os

import datetime as dt
import pandas as pd

from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.als_files import load_bin_als_files
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_mstd_figure
from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure

if __name__ == '__main__':
    rootdir = select_dir_path()

    fish_tracks_bin = load_bin_als_files(rootdir, suffix="*als_30m.csv")

    fish_IDs = fish_tracks_bin['FishID'].unique()
    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s, \
    change_times_d, change_times_m, change_times_datetime, change_times_unit = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # convert ts to datetime
    fish_tracks_bin['ts'] = pd.to_datetime(fish_tracks_bin['ts'])

    # plot weekly plot
    plot_speed_30m_mstd_figure(rootdir, fish_tracks_bin, change_times_d, ylim_max=85)

    # plot daily plot
    all_species = fish_tracks_bin['species'].unique()
    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts']]
        sp_spd = spd.pivot(columns='FishID', values='speed_mm', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_spd['time_of_day'] = sp_spd.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_ave = sp_spd.groupby('time_of_day').mean()
        sp_spd_ave_std = sp_spd_ave.std(axis=1)

        # make the plots
        daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit)
        daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=85)