import warnings
import os

import datetime as dt
import pandas as pd
import numpy as np

from cichlidanalysis.utils.timings import load_timings, load_timings_14_8
from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.als_files import load_bin_als_files
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_mstd_figure_info, weekly_individual_figure
from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure, daily_ave_spd_figure_night_centred, daily_ave_spd_figure_sex
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
    # plot_speed_30m_mstd_figure_info(rootdir, fish_tracks_bin, change_times_d, diel_guilds, cichlid_meta, temporal_col,
    #                                 ylim_max=60)

    # plot all individuals for each species
    # weekly_individual_figure(rootdir, 'speed_mm', fish_tracks_bin, change_times_m, bin_size_min=bin_size_min)

    # plot daily plot separated by sex
    all_species = fish_tracks_bin['species'].unique()
    for species_f in all_species:
        # ### speed ###
        spd = fish_tracks_bin[fish_tracks_bin.species == species_f][['speed_mm', 'FishID', 'ts', 'sex']]
        spd_male = spd.loc[spd.sex == 'm']
        spd_female = spd.loc[spd.sex == 'f']
        spd_unknown = spd.loc[spd.sex == 'u']

        sp_spd_all = spd.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_male = spd_male.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_female = spd_female.pivot(columns='FishID', values='speed_mm', index='ts')
        sp_spd_unknown = spd_unknown.pivot(columns='FishID', values='speed_mm', index='ts')

        # get time of day so that the same tod for each fish can be averaged
        sp_spd_all['time_of_day'] = sp_spd_all.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_male['time_of_day'] = sp_spd_male.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_female['time_of_day'] = sp_spd_female.apply(lambda row: str(row.name)[11:16], axis=1)
        sp_spd_unknown['time_of_day'] = sp_spd_unknown.apply(lambda row: str(row.name)[11:16], axis=1)


        def get_mean_std(sp_spd_all, sp_spd_subset):
            if not sp_spd_subset.empty:
                sp_spd_subset_ave = sp_spd_subset.groupby('time_of_day').mean()
                daily_speed_subset = sp_spd_subset_ave.mean(axis=1)
                sp_spd_subset_ave_std = sp_spd_subset_ave.std(axis=1)
            else:
                sp_spd_ave_all = sp_spd_all.groupby('time_of_day').mean()
                daily_speed_subset = pd.Series(data=np.nan, index=sp_spd_ave_all.index)
                sp_spd_ave_std_subset = pd.Series(data=np.nan, index=sp_spd_ave_all.index)
            return daily_speed_subset, sp_spd_ave_std_subset


        if not sp_spd_male.empty:
            sp_spd_ave_male = sp_spd_male.groupby('time_of_day').mean()
            sp_spd_ave_std_male = sp_spd_ave_male.std(axis=1)
            daily_speed_male = sp_spd_ave_male.mean(axis=1)
        else:
            sp_spd_ave_all = sp_spd_all.groupby('time_of_day').mean()
            sp_spd_ave_male = pd.Series(data=np.nan, index=sp_spd_ave_all.index)
            sp_spd_ave_std_male = pd.Series(data=np.nan, index=sp_spd_ave_all.index)

        if not sp_spd_female.empty:
            sp_spd_ave_female = sp_spd_female.groupby('time_of_day').mean()
            sp_spd_ave_std_female = sp_spd_ave_female.std(axis=1)
        else:
            sp_spd_ave_all = sp_spd_all.groupby('time_of_day').mean()
            sp_spd_ave_female = pd.Series(data=np.nan, index=sp_spd_ave_all.index)
            sp_spd_ave_std_female = pd.Series(data=np.nan, index=sp_spd_ave_all.index)

        if not sp_spd_unknown.empty:
            sp_spd_ave_unknown = sp_spd_unknown.groupby('time_of_day').mean()
            sp_spd_ave_std_unknown = sp_spd_ave_unknown.std(axis=1)
        else:
            sp_spd_ave_all = sp_spd_all.groupby('time_of_day').mean()
            sp_spd_ave_unknown = pd.Series(data=np.nan, index=sp_spd_ave_all.index)
            sp_spd_ave_std_unknown = pd.Series(data=np.nan, index=sp_spd_ave_all.index)

        sp_spd_ave = pd.DataFrame({
            'male': sp_spd_ave_male,
            'female': sp_spd_ave_female,
            'unknown': sp_spd_ave_unknown
        })

        daily_ave_spd_figure_sex(rootdir, sp_spd_ave, sp_spd_ave_std_male, species_f, change_times_unit, ymax=100)

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
        daily_ave_spd_figure_night_centred(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit)
        # daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit)
        # daily_ave_spd_figure(rootdir, sp_spd_ave, sp_spd_ave_std, species_f, change_times_unit, ymax=100)

