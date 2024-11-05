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
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.meta import load_meta_files
from cichlidanalysis.io.als_files import load_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import add_col, threshold_data, remove_cols
from cichlidanalysis.analysis.behavioural_state import define_rest
from cichlidanalysis.analysis.processing import smooth_speed



# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def combine_binning(rootdir, binning_m=30):
    # ### Movement moving/not-moving use 15mm/s threshold ####
    MOVE_THRESH = 15

    # ### Behavioural state - calculated from Movement ###
    TIME_WINDOW_SEC = 60
    FRACTION_THRESH = 0.05

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
    fish_tracks_bin = fish_tracks.groupby('FishID').resample((str(binning_m)+'T'), on='ts').mean()
    fish_tracks_bin.reset_index(inplace=True)
    print("calculated resampled {}min data".format(binning_m))

    # add new column with Day or Night
    t2 = time.time()
    fish_tracks_bin = fish_tracks_bin.dropna()  # occaisionally have NaTs in ts, this removes them.
    fish_tracks_bin.loc[:, 'time_of_day_m'] = fish_tracks_bin.ts.apply(
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
    print("Finished saving out {}min data".format(binning_m))

    # get speed stats and save out individual fish plots
    for fish in fish_tracks.FishID.unique():
        bins = np.arange(0, 1000, 5)  # +5 to include the max value

        plt.figure(figsize=(10, 6))
        plt.hist(fish_tracks.loc[fish_tracks.FishID == fish, 'speed_mm'], bins=bins, edgecolor='black') #, log=True
        plt.xlabel('Speed mm/s')
        plt.ylabel('Frequency')
        plt.xlim([0, 250])
        plt.title('Histogram of Speed')
        plt.axvline(MOVE_THRESH, color='r', label='15mm/s')
        plt.axvline(0.25*meta.T.fish_length_mm[fish], color='k', linestyle='--', label='0.25 body lengths')
        plt.legend()
        plt.savefig(os.path.join(rootdir, "speed_hist_{}_{}.pdf".format(all_species[0].replace(' ', '-'), fish)), dpi=350)

    # save out speed percentiles
    percentile_values = fish_tracks.groupby('FishID')['speed_mm'].quantile([0.50, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]).unstack()
    percentile_values_meta = pd.concat([percentile_values, meta.T], axis=1)
    percentile_values_meta.to_csv(os.path.join(rootdir, "{}_spd_percentiles.csv".format(all_species[0])))

    return


def combine_smoothing(rootdir, smoothing_win_f, down_sample_bin):

    t0 = time.time()
    fish_tracks = load_als_files(rootdir)
    t1 = time.time()
    print("time to load tracks {:.0f} sec".format(t1 - t0))

    meta = load_meta_files(rootdir)

    # get each fish ID
    fish_IDs = fish_tracks['FishID'].unique()

    # data gets heavy so remove what is not necessary
    fish_tracks = remove_cols(fish_tracks, ['y_nt', 'x_nt', 'tv_ns'])

    first = True
    for fish in fish_tracks.FishID.unique():
        spd_fish_sm = pd.Series(smooth_speed(fish_tracks.loc[fish_tracks.FishID == fish, "speed_mm"],
                                             win_size=smoothing_win_f)[:, 0], name='spd_mm_sm')

        fish_tracks_sm = fish_tracks.loc[fish_tracks.FishID == fish, "ts"].to_frame().join(spd_fish_sm)
        fish_tracks_sm_ds_fish = fish_tracks_sm.loc[::600].reset_index(drop=True)

        fish_tracks_sm_ds_fish['FishID'] = fish

        if first:
            fish_tracks_sm_ds = fish_tracks_sm_ds_fish
            first = False
        else:
            fish_tracks_sm_ds = pd.concat([fish_tracks_sm_ds, fish_tracks_sm_ds_fish])

        # time_s = 30*60
        # time_e = 32*60
        # # plt.plot(fish_tracks.loc[time_s*10*60:time_e*10*60, 'ts'], fish_tracks.loc[time_s*10*60:time_e*10*60, 'speed_mm'], c='orange')
        # plt.plot(fish_tracks_sm.loc[time_s*10*60:time_e*10*60, 'ts'], fish_tracks_sm.loc[time_s*10*60:time_e*10*60, 'spd_mm_sm'], c='blue')
        # # plt.plot(fish_tracks_sm_1min.loc[time_s*10*60:time_e*10*60, 'ts'], fish_tracks_sm_1min.loc[time_s*10*60:time_e*10*60, 'spd_mm_sm'], c='blue')
        # plt.plot(fish_tracks_sm_ds.loc[time_s*10*60:time_e*10*60, 'ts'], fish_tracks_sm_ds.loc[time_s*10*60:time_e*10*60, 'spd_mm_sm'], c='yellow')
        # plt.plot(fish_tracks_bin_f.loc[time_s/30:time_e/30, 'ts'], fish_tracks_bin_f.loc[time_s/30:time_e/30, 'speed_mm'], c='green')
        # plt.plot(fish_tracks_bin_10.loc[time_s/10:time_e/10, 'ts'], fish_tracks_bin_10.loc[time_s/10:time_e/10, 'speed_mm'], c='purple')
        # # plt.plot(fish_tracks_bin_1.loc[time_s/1:time_e/1, 'ts'], fish_tracks_bin_1.loc[time_s/1:time_e/1, 'speed_mm'], c='red')
        # plt.savefig(os.path.join(rootdir, "test_smoothing.png"))
        # plt.close()

    # add back 'species', 'sex'
    for col_name in ['species', 'sex']:
        add_col(fish_tracks_sm_ds, col_name, fish_IDs, meta)
    all_species = fish_tracks_sm_ds['species'].unique()

    # save out downsampled als
    for species in all_species:
        fish_tracks_sm_ds.to_csv(os.path.join(rootdir, "{}_sm{}_als_{}m.csv".format(species, smoothing_win_f, down_sample_bin)))
    print("Finished saving out double smoothed down-sampled {}min data".format(down_sample_bin))
    return


if __name__ == '__main__':

    rootdir = select_dir_path()

    combine_binning(rootdir, binning_m=30)

    # fps = 10
    # smoothing_win_f = 60*10*fps
    # down_sample_bin_min = 5
    # combine_smoothing(rootdir, smoothing_win_f, down_sample_bin_min)


