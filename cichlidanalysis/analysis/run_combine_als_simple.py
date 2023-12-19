############################
# This module loads als and meta data of individual fish and plots the following for each species:
# speed_mm (30m bins, daily ave) for each fish (lines single and average as well as heatmap)
# x,y position (binned day/night, and average day/night)
# fraction movement/not movement
# fraction rest/non-rest
# bout structure (movement and rest, bout fraction in 30min bins, bouts D/N over days)

import warnings
import time
import os

import numpy as np

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.meta import load_meta_files
from cichlidanalysis.io.als_files import load_als_files
from cichlidanalysis.io.io_feature_vector import create_fv1, create_fv2
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import add_col, threshold_data, remove_cols
from cichlidanalysis.analysis.bouts import find_bouts_input
from cichlidanalysis.analysis.behavioural_state import define_rest, plotting_clustering_states
from cichlidanalysis.plotting.position_plots import spd_vs_y, plot_position_maps
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_individuals, plot_speed_30m_mstd, plot_speed_30m_sex, \
    plot_speed_30m_mstd_figure
from cichlidanalysis.plotting.movement_plots import plot_movement_30m_individuals, plot_movement_30m_mstd, \
    plot_bout_lengths_dn_move, plot_movement_30m_sex
from cichlidanalysis.plotting.daily_plots import plot_daily
from cichlidanalysis.plotting.rest_plots import plot_rest_ind, plot_rest_mstd, plot_rest_bout_lengths_dn, plot_rest_sex

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # ### Movement moving/not-moving use 15mm/s threshold ####
    MOVE_THRESH = 15

    # ### Behavioural state - calculated from Movement ###
    TIME_WINDOW_SEC = 60
    FRACTION_THRESH = 0.05

    binning_m = 5

    rootdir = select_dir_path()

    t0 = time.time()
    fish_tracks = load_als_files(rootdir)
    t1 = time.time()
    print("time to load tracks {:.0f} sec".format(t1 - t0))

    meta = load_meta_files(rootdir)
    metat = meta.transpose()

    # get each fish ID
    fish_IDs = fish_tracks['FishID'].unique()

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s, \
    change_times_d, change_times_m, change_times_datetime, change_times_unit = \
        load_timings(fish_tracks[fish_tracks.FishID == fish_IDs[0]].shape[0])

    fish_tracks['movement'] = np.nan
    for fish in fish_IDs:
        # threshold the speed_mm with 15mm/s
        fish_tracks.loc[(fish_tracks.FishID == fish), 'movement'] = threshold_data(
            fish_tracks.loc[(fish_tracks.FishID == fish), "speed_mm"], MOVE_THRESH)

    # define behave states
    fish_tracks = define_rest(fish_tracks, TIME_WINDOW_SEC, FRACTION_THRESH)

    # #### x,y position (binned day/night, and average day/night) #####
    # normalising positional data
    horizontal_pos = fish_tracks.pivot(columns="FishID", values="x_nt")
    vertical_pos = fish_tracks.pivot(columns="FishID", values="y_nt")

    # scale each fish by min/max
    horizontal_pos -= horizontal_pos.min()
    horizontal_pos /= horizontal_pos.max()
    vertical_pos -= vertical_pos.min()
    vertical_pos /= vertical_pos.max()
    # flip Y axis
    vertical_pos = abs(1 - vertical_pos)

    # put this data back into fish_tracks
    fish_tracks['vertical_pos'] = np.nan
    fish_tracks['horizontal_pos'] = np.nan
    for fish in fish_IDs:
        fish_tracks.loc[fish_tracks.FishID == fish, 'vertical_pos'] = vertical_pos.loc[:, fish]
        fish_tracks.loc[fish_tracks.FishID == fish, 'horizontal_pos'] = horizontal_pos.loc[:, fish]
    print("added vertical and horizontal position columns")

    # data gets heavy so remove what is not necessary
    fish_tracks = remove_cols(fish_tracks, ['y_nt', 'x_nt', 'tv_ns'])

    # resample data
    fish_tracks_bin = fish_tracks.groupby('FishID').resample((binning_m+'T'), on='ts').mean()
    fish_tracks_bin.reset_index(inplace=True)
    print("calculated resampled 30min data")

    # add new column with Day or Night
    t2 = time.time()
    fish_tracks_bin = fish_tracks_bin.dropna()  # occaisionally have NaTs in ts, this removes them.
    fish_tracks_bin['time_of_day_m'] = fish_tracks_bin.ts.apply(
        lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    t3 = time.time()
    print("time to add time_of_day tracks {:.0f} sec".format(t3 - t2))

    fish_tracks_bin['daynight'] = "d"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m > change_times_m[3], 'daynight'] = "n"
    print("added night and day column")

    # add back 'species', 'sex'
    for col_name in ['species', 'sex']:
        add_col(fish_tracks_bin, col_name, fish_IDs, meta)
    all_species = fish_tracks_bin['species'].unique()

    fish_tracks_bin['daynight'] = "d"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_tracks_bin.loc[fish_tracks_bin.time_of_day_m > change_times_m[3], 'daynight'] = "n"
    print("Finished adding bin species and daynight")


    # save out downsampled als
    for species in all_species:
        fish_tracks_bin.to_csv(os.path.join(rootdir, "{}_als_{}m.csv".format(species, binning_m)))
    print("Finished saving out 30min data")

    # # feature vectors: for each fish readout vector of feature values
    # create_fv1(all_species, fish_IDs, fish_tracks, metat, rootdir)
    # create_fv2(all_species, fish_tracks, fish_bouts_move, fish_bouts_rest, fish_IDs, metat, fish_tracks_bin, rootdir)

fish_tracks_bin.columns
Index(['FishID', 'ts', 'speed_mm', 'time_of_day_m', 'movement', 'rest',
       'vertical_pos', 'horizontal_pos', 'species', 'sex', 'daynight'],
      dtype='object')