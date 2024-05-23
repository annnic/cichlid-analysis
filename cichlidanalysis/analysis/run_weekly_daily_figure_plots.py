import warnings
import os

import datetime as dt
import pandas as pd

from cichlidanalysis.utils.timings import load_timings, load_timings_14_8
from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.als_files import load_bin_als_files
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_mstd_figure_info, weekly_individual_figure
from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure
from cichlidanalysis.io.io_ecological_measures import get_meta_paths

if __name__ == '__main__':
    # plots the weekly mean +- stdev plots for extended figure 1+2
    rootdir = select_dir_path()

    bin_size_min = 30
    fish_tracks_bin = load_bin_als_files(rootdir, suffix="*als_{}m.csv".format(bin_size_min))

    table_1 = pd.read_csv(os.path.join(rootdir, "table_1.csv"), sep=',')
    diel_guilds = pd.read_csv(os.path.join(rootdir, "diel_guilds.csv"), sep=',')
    temporal_col = {'Crepuscular': '#26D97A', 'Nocturnal': '#40A9BF', 'Diurnal': '#CED926', 'Cathemeral': '#737F8C'}

    _, cichlid_meta_path = get_meta_paths()
    cichlid_meta = pd.read_csv(cichlid_meta_path)

    fish_IDs = fish_tracks_bin['FishID'].unique()
    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s, \
    change_times_d, change_times_m, change_times_datetime, change_times_unit = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # convert ts to datetime
    fish_tracks_bin['ts'] = pd.to_datetime(fish_tracks_bin['ts'])

    # plot weekly plot
    plot_speed_30m_mstd_figure_info(rootdir, fish_tracks_bin, change_times_d, diel_guilds, cichlid_meta, temporal_col,
                                    ylim_max=60)

    # plot all individuals for each species
    weekly_individual_figure(rootdir, 'speed_mm', fish_tracks_bin, change_times_m, bin_size_min=bin_size_min)

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
        daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=100)
