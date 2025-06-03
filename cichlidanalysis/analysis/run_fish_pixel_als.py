import copy
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)
import matplotlib
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.gridspec as grid_spec
import datetime as dt
from datetime import timedelta
import matplotlib.patches as patches

from cichlidanalysis.io.meta import load_yaml
from cichlidanalysis.io.tracks import extract_tracks_from_fld, adjust_old_time_ns
from cichlidanalysis.utils.timings import output_timings, get_start_time_of_video, set_time_vector
from cichlidanalysis.analysis.processing import interpolate_nan_streches, remove_high_spd_xy, smooth_speed, neg_values
from cichlidanalysis.plotting.single_plots import filled_plot, plot_hist_2, image_minmax, sec_axis_h
from cichlidanalysis.io.get_file_folder_paths import select_dir_path, select_top_folder_path
from cichlidanalysis.utils.species_names import get_roi_from_fish_id
from cichlidanalysis.plotting.single_plots import fill_plot_ts


# def pixels_add_timestamps(rootdir, orig_files_df, pixel_files_df):
#     # add old timestamps to newly re-tracked data
#     for orig_n, orig_file in enumerate(orig_files_df.file_name):
#         for new_f in pixel_files_df.file_name:
#             if orig_file[0:18] == new_f[0:18]:
#                 print("updating timestamps of {} and adding exclude tag to {}".format(new_f, orig_file))
#                 update_csvs_pixels(os.path.join(rootdir, orig_file), os.path.join(rootdir, new_f))
#     return

def plot_pixel_30m(rootdir, pixel_track_df_30min, change_times_d, ylim_max=60):
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    date_form = DateFormatter("%H")

    # adjusted_datetime = pixel_track_df_30min.tv - pd.DateOffset(months=2, days=10)

    plt.figure(figsize=(2, 1))
    ax = sns.lineplot(x=pixel_track_df_30min.tv, y=pixel_track_df_30min.d_pixels, linewidth=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(date_form)

    fill_plot_ts(ax, change_times_d, pixel_track_df_30min.tv)
    plt.xlabel("Time (h)", fontsize=SMALLEST_SIZE)
    plt.ylabel("delta pixels", fontsize=SMALLEST_SIZE)

    # Decrease the offset for tick labels on all axes
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5

    # Adjust the offset for tick labels on all axes
    ax.tick_params(axis='x', pad=0.5, length=2)
    ax.tick_params(axis='y', pad=0.5, length=2)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "pixel_D.pdf"), dpi=350)
    plt.close()
    return


    fig2, ax2 = filled_plot(pixel_track_df_30min.ts, pixel_track_df_30min.d_pixels, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.ylabel("average y position")
    plt.title("Y position_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))
    plt.savefig(os.path.join(rootdir, "{0}_Y-position.png".format(FISH_ID)))
    return


def daily_ave_pixel_figure(rootdir, pixel_track_df_30min, change_times_m):
    """ speed_mm (30m bins daily average) for each fish (individual lines)

    :param rootdir:
    :param pixel_track_df_30min:
    :param change_times_unit:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    date_form = DateFormatter("%M")

    # get time of day so that the same tod for each fish can be averaged
    pixel_track_df_30min = pixel_track_df_30min.dropna()  # occaisionally have NaTs in ts, this removes them.
    mins = pixel_track_df_30min.tv.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    pixel_track_df_30min['time_of_day_m'] = mins

    pixel_track_df_30min_ave = pixel_track_df_30min.groupby('time_of_day_m').mean()
    pixel_track_df_30min_std = pixel_track_df_30min.groupby('time_of_day_m').std()

    plt.figure(figsize=(1.2, 1.2))
    ax = sns.lineplot(x=pixel_track_df_30min_ave.index, y=(pixel_track_df_30min_ave.d_pixels + pixel_track_df_30min_std.d_pixels),
                      color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=pixel_track_df_30min_ave.index, y=(pixel_track_df_30min_ave.d_pixels - pixel_track_df_30min_std.d_pixels),
                      color='lightgrey', linewidth=0.5)
    ax = sns.lineplot(x=pixel_track_df_30min_ave.index, y=(pixel_track_df_30min_ave.d_pixels), linewidth=0.5)

    ax.axvspan(0, change_times_m[0], color='lightblue', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_m[0], change_times_m[1], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_m[2], change_times_m[3], color='wheat', alpha=0.5, linewidth=0)
    ax.axvspan(change_times_m[3], 60*24, color='lightblue', alpha=0.5, linewidth=0)
    ax.set_xlim([0, 60*24])
    plt.xlabel("Time (h)", fontsize=SMALL_SIZE)
    plt.ylabel("pixel delta", fontsize=SMALL_SIZE)

    # Decrease the offset for tick labels on all axes
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5

    # Adjust the offset for tick labels on all axes
    ax.tick_params(axis='x', pad=0.5, length=2)
    ax.tick_params(axis='y', pad=0.5, length=2)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_major_locator(MultipleLocator(60*6))
    # ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "pixels_30min_daily_ave.pdf"))
    plt.close()
    return


def full_analysis(rootdir):
    """ analyses the data of one fish's recording

    :param rootdir:
    :return:
    """
    NUM_DAYS = 7
    MIN_BINS = 30

    FILE_PATH_PARTS = os.path.split(rootdir)
    config = load_yaml(FILE_PATH_PARTS[0], "config")
    meta = load_yaml(rootdir, "meta_data")
    FISH_ID = FILE_PATH_PARTS[1]
    MOVE_THRESH = 15

    file_ending = get_roi_from_fish_id(FISH_ID)

    # get all files and their movie numbers
    os.chdir(rootdir)
    all_files = glob.glob("*.csv")
    all_files_df = pd.DataFrame(all_files, columns=['file_name'])
    all_files_df.file_name.str.split('_',  expand=True)
    all_files_df["movie_n"] = all_files_df.file_name.str.split('_',  expand=True).iloc[:, 1]

    indices = []
    for row_idx, value in enumerate(all_files_df.file_name):
        if 'pixel' in str(value):
             indices.append(row_idx)

    pixel_files_df = all_files_df.iloc[indices, :].sort_values(by='movie_n')
    orig_files_df = all_files_df.drop(indices).sort_values(by='movie_n')

    # load tracks
    pixel_track = np.empty([0, 2])
    for file in pixel_files_df.file_name:
        print(file)
        csv_file_path = os.path.join(rootdir, file)
        pixel_track_single = pd.read_csv(csv_file_path, names=['ts', 'd_pixels'], header=None)
        # change ts to datetime
        pixel_track_single['ts'] = pd.to_datetime(pixel_track_single['ts'], unit='ns')
        # change d_pixels to int
        pixel_track_single['d_pixels'] = pixel_track_single['d_pixels'].astype(float)
        if len(pixel_track_single) > 0:
           pixel_track = np.append(pixel_track, pixel_track_single, axis=0)
    pixel_track_df = pd.DataFrame(pixel_track, columns=['ts', 'd_pixels'])
    pixel_track_df['d_pixels'] = pixel_track_df['d_pixels'].astype(float)

    # track_full = np.empty([0, 4])
    # for file in orig_files_df.file_name:
    #     print(file)
    #     csv_file_path = os.path.join(rootdir, file)
    #     track_single = pd.read_csv(csv_file_path, names=['ts', 'x', 'y', 'area'], header=None)
    #     # change ts to datetime
    #     track_single['ts'] = pd.to_datetime(track_single['ts'], unit='ns')
    #     if len(track_single) > 0:
    #        track_full = np.append(track_full, track_single, axis=0)
    # track_full_df = pd.DataFrame(track_full, columns=['ts', 'x', 'y', 'area'])

    # for old recordings update time (subtract 30min) - need to check if this works on this dtype, this is automatically done in load_als_files
    # track_full_df.iloc[:, 0] = adjust_old_time_ns(FISH_ID, track_full_df.iloc[:, 0])
    # pixel_track_df.iloc[:, 0] = adjust_old_time_ns(FISH_ID, pixel_track_df.iloc[:, 0])

    # get starting time of video
    video_start_total_sec = get_start_time_of_video(rootdir)

    # set sunrise, day, sunset, night times (ns, s, m, h) and set day length in ns, s and d
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, change_times_datetime, change_times_unit = output_timings()

    tv = set_time_vector(pixel_track_df.ts.to_numpy().reshape(-1, 1), video_start_total_sec, config)
    tv = tv.astype('datetime64[ns]')
    pixel_track_df.loc[:, 'tv'] = tv[0:pixel_track_df.shape[0]]

    # # correct to seconds
    # NS_IN_SECONDS = 10 ** 9
    # tv_sec = tv / NS_IN_SECONDS
    # tv_24h_sec = tv / NS_IN_SECONDS

    if int(FISH_ID[4:12]) < 20201127:
        thirty_min_ns = 30 * 60 * 1000000000
        adjusted_day_ns = day_ns - thirty_min_ns
        print("old recording from before 20201127 so adjusting back time before saving out als")
    else:
        adjusted_day_ns = day_ns
    adjusted_day_ns = np.datetime64(adjusted_day_ns, 'ns')
    tv = tv.astype('datetime64[ns]')

    # Drop rows where 'tv' is below adjusted_day_ns (first midnight)
    pixel_track_df = pixel_track_df[pixel_track_df['tv'] >= adjusted_day_ns]

    pixel_track_df_30min = pixel_track_df.resample('30T', on='tv').mean()
    pixel_track_df_30min = pixel_track_df_30min.reset_index()

    plot_pixel_30m(rootdir, pixel_track_df_30min, change_times_d)

    # get daily average
    daily_ave_pixel_figure(rootdir, pixel_track_df_30min, change_times_m)

    return


if __name__ == '__main__':
    analyse_multiple_folders = 'm'
    while analyse_multiple_folders not in {'y', 'n'}:
        analyse_multiple_folders = input("Analyse multiple folders (ROIs) (y) or only one ROI (n)?: \n")

    if analyse_multiple_folders == 'n':
        rootdir = select_dir_path()
        full_analysis(rootdir)
    else:
        topdir = select_top_folder_path()
        list_subfolders_with_paths = [f.path for f in os.scandir(topdir) if f.is_dir()]

        for camera_folder in list_subfolders_with_paths:
            list_subsubfolders_with_paths = [f.path for f in os.scandir(camera_folder) if f.is_dir()]
            # for skipping folders with lights
            list_subsubfolders_with_paths_without_lights = []
            for i in list_subsubfolders_with_paths:
                if i[-3:] != '_sl':
                    list_subsubfolders_with_paths_without_lights.append(i)

            for roi_folder in list_subsubfolders_with_paths_without_lights:
                if roi_folder.find('EXCLUDE') == -1:
                    full_analysis(roi_folder)
