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
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import threshold_data, remove_cols
from cichlidanalysis.analysis.behavioural_state import define_rest

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

    # get the vertical position for rest and non-rest periods for each fish
    # for each indidviudal, rest yes/no, vp
    rest_vp = fish_tracks.loc[:, ['FishID', "rest", 'vertical_pos']].groupby(by=['FishID', "rest"]).mean().reset_index()

    species = fish_IDs[0].split('_')[3]
    rest_vp.to_csv(os.path.join(rootdir, "rest_vp_{}.csv".format(species)))