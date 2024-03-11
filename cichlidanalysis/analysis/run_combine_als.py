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

    # add new column with Day or Night
    t2 = time.time()
    fish_tracks = fish_tracks.dropna()  # occaisionally have NaTs in ts, this removes them.
    fish_tracks['time_of_day_m'] = fish_tracks.ts.apply(
        lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    t3 = time.time()
    print("time to add time_of_day tracks {:.0f} sec".format(t3 - t2))

    fish_tracks['daynight'] = "d"
    fish_tracks.loc[fish_tracks.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_tracks.loc[fish_tracks.time_of_day_m > change_times_m[3], 'daynight'] = "n"
    print("added night and day column")

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

    # get the vertical position for

    # resample data
    fish_tracks_30m = fish_tracks.groupby('FishID').resample('30T', on='ts').mean()
    fish_tracks_30m.reset_index(inplace=True)
    print("calculated resampled 30min data")

    # add back 'species', 'sex'
    for col_name in ['species', 'sex']:
        add_col(fish_tracks_30m, col_name, fish_IDs, meta)
    all_species = fish_tracks_30m['species'].unique()

    fish_tracks_30m['daynight'] = "d"
    fish_tracks_30m.loc[fish_tracks_30m.time_of_day_m < change_times_m[0], 'daynight'] = "n"
    fish_tracks_30m.loc[fish_tracks_30m.time_of_day_m > change_times_m[3], 'daynight'] = "n"
    print("Finished adding 30min species and daynight")

    # define bouts for movement
    fish_bouts_move = find_bouts_input(fish_tracks, change_times_m, measure="movement")
    print("Defined bouts for movement")

    # define bouts for rest
    fish_bouts_rest = find_bouts_input(fish_tracks, change_times_m, measure="rest")
    print("Defined bouts for rest")

    # ### plotting ### #
    # ### SPEED ###
    # speed_mm (30m bins) for each fish (individual lines)
    plot_speed_30m_individuals(rootdir, fish_tracks_30m, change_times_d)

    # speed_mm (30m bins) for each species (mean  +- std)
    plot_speed_30m_mstd_figure(rootdir, fish_tracks_30m, change_times_d)
    plot_speed_30m_mstd(rootdir, fish_tracks_30m, change_times_d)
    plot_speed_30m_sex(rootdir, fish_tracks_30m, change_times_d)

    # plotting_clustering_states(rootdir, fish_tracks,  resample_units=['1S', '3S', '10S', '30S', '2T', '10T'])
    print("Finished speed plots")

    # ### MOVEMENT ###
    # movement for each fish (individual lines)
    plot_movement_30m_individuals(rootdir, fish_tracks_30m, change_times_d, MOVE_THRESH)

    # movement (30m bins) for each species (mean  +- std)
    plot_movement_30m_mstd(rootdir, fish_tracks_30m, change_times_d, MOVE_THRESH)
    plot_movement_30m_sex(rootdir, fish_tracks_30m, change_times_d, MOVE_THRESH)
    plot_bout_lengths_dn_move(fish_bouts_move, rootdir)
    print("Finished movement plots")

    # ### POSITION ###
    # ##### x,y position (binned day/night, and average day/night) #####
    plot_position_maps(meta, fish_tracks, rootdir)

    # speed vs Y position, for each fish, for combine fish of species, separated between day and night
    spd_vs_y(meta, fish_tracks_30m, fish_IDs, rootdir)
    print("Finished position plots")

    # ### REST ###
    # rest (30m bins) for each fish (individual lines)
    plot_rest_ind(rootdir, fish_tracks_30m, change_times_d, FRACTION_THRESH, TIME_WINDOW_SEC, "30m")

    # rest (30m bins) for each species (mean  +- std)
    plot_rest_mstd(rootdir, fish_tracks_30m, change_times_d, "30m")
    plot_rest_sex(rootdir, fish_tracks_30m, change_times_d, FRACTION_THRESH, TIME_WINDOW_SEC, "30m")
    plot_rest_bout_lengths_dn(fish_bouts_rest, rootdir)
    print("Finished rest plots")

    # get daily average
    plot_daily(fish_tracks_30m, change_times_unit, rootdir)

    # save out downsampled als
    for species in all_species:
        fish_tracks_30m.to_csv(os.path.join(rootdir, "{}_als_30m.csv".format(species)))
    print("Finished saving out 30min data")

    # feature vectors: for each fish readout vector of feature values
    create_fv1(all_species, fish_IDs, fish_tracks, metat, rootdir)
    create_fv2(all_species, fish_tracks, fish_bouts_move, fish_bouts_rest, fish_IDs, metat, fish_tracks_30m, rootdir)

