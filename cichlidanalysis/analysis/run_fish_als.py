import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cichlidanalysis.io.meta import load_yaml
from cichlidanalysis.io.tracks import extract_tracks_from_fld, adjust_old_time_ns
from cichlidanalysis.utils.timings import output_timings, get_start_time_of_video, set_time_vector
from cichlidanalysis.analysis.processing import interpolate_nan_streches, remove_high_spd_xy, smooth_speed, neg_values
from cichlidanalysis.plotting.single_plots import filled_plot, plot_hist_2, image_minmax, sec_axis_h
from cichlidanalysis.io.get_file_folder_paths import select_dir_path, select_top_folder_path
from cichlidanalysis.utils.species_names import get_roi_from_fish_id


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

    # load tracks
    track_full, speed_full = extract_tracks_from_fld(rootdir, file_ending)

    # for old recordings update time (subtract 30min)
    track_full[:, 0] = adjust_old_time_ns(FISH_ID, track_full[:, 0])

    # get starting time of video
    video_start_total_sec = get_start_time_of_video(rootdir)

    # set sunrise, day, sunset, night times (ns, s, m, h) and set day length in ns, s and d
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, _, _ = output_timings()

    # this is not time adjusted
    tv = set_time_vector(track_full, video_start_total_sec, config)

    # correct to seconds
    NS_IN_SECONDS = 10 ** 9
    tv_sec = tv / NS_IN_SECONDS
    tv_24h_sec = tv / NS_IN_SECONDS

    # get time vector with 24h time
    for day in range(NUM_DAYS):
        tv_24h_sec[np.where(tv_24h_sec > day_ns / NS_IN_SECONDS)] -= day_ns / NS_IN_SECONDS

    # interpolate between NaN stretches
    x_n = interpolate_nan_streches(track_full[:, 1])
    y_n = interpolate_nan_streches(track_full[:, 2])

    # replace bad track NaNs (-1) -> these are manually defined as artifacts
    x_n[np.where(x_n == -1)] = np.nan
    y_n[np.where(y_n == -1)] = np.nan

    # find displacement
    speed_full_i = np.sqrt(np.diff(x_n) ** 2 + np.diff(y_n) ** 2)
    speed_t, x_nt, y_nt = remove_high_spd_xy(speed_full_i, x_n, y_n)

    speed_sm = smooth_speed(speed_t, win_size=5)
    speed_sm_mm = speed_sm * config["mm_per_pixel"]
    speed_sm_mm_ps = speed_sm_mm * config['fps']
    speed_sm_tbl = speed_sm_mm / meta["fish_length_mm"]
    speed_sm_tbl_ps = speed_sm_tbl * config['fps']

    # smoothing on coordinates
    fig1, ax1 = plt.subplots()
    plt.hist(np.diff(x_nt), 1000)
    plt.yscale('log')
    plt.xlabel("pixels")
    plt.ylabel("frequency")
    plt.title("X_{0}".format(meta["species"]))
    plt.savefig(os.path.join(rootdir, "{0}_X_jumps.png".format(FISH_ID)))

    fig1, ax1 = plt.subplots()
    plt.hist(np.diff(y_nt), 1000)
    plt.yscale('log')
    plt.xlabel("pixels")
    plt.ylabel("frequency")
    plt.title("Y_{0}".format(meta["species"]))
    plt.savefig(os.path.join(rootdir, "{0}_Y-jumps.png".format(FISH_ID)))
    plt.close()

    y_sm = smooth_speed(y_nt, win_size=5)
    smooth_win = 10 * 60 * MIN_BINS
    y_bin = smooth_speed(y_sm, win_size=smooth_win)

    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, y_bin[0:-1], change_times_h, day_ns / 10 ** 9 / 60 / 60)

    plt.ylabel("average y position")
    plt.title("Y position_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))
    plt.savefig(os.path.join(rootdir, "{0}_Y-position.png".format(FISH_ID)))

    # area
    area_sm = smooth_speed(track_full[0:-1, 3], win_size=5)
    area_bin = smooth_speed(area_sm, win_size=smooth_win)

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, area_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.xlabel("Time (h)")
    plt.ylabel("average area size")
    plt.title("Area_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))

    # split data into day and  night
    tv_night = np.empty([0, 0])
    speed_sm_night = np.empty([0, 0])

    tv_night = np.append(tv_night, tv_24h_sec[np.where(change_times_s[0] > tv_24h_sec)])
    tv_night = np.append(tv_night, tv[np.where(tv_24h_sec[0:-1] > change_times_s[3])])

    speed_sm_night = np.append(speed_sm_night, speed_sm_mm_ps[np.where(change_times_s[0] > tv_24h_sec[0:-1]), 0])
    speed_sm_night = np.append(speed_sm_night, speed_sm_mm_ps[np.where(tv_24h_sec[0:-1] > change_times_s[3]), 0])

    tv_day = np.empty([0, 0])
    speed_sm_day = np.empty([0, 0])

    tv_day = np.append(tv_day, tv[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                           (tv_24h_sec[0:-1] < change_times_s[3]))])
    speed_sm_day = np.append(speed_sm_day, speed_sm_mm_ps[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                                             (tv_24h_sec[0:-1] < change_times_s[3])), 0])

    # plot speed distributions
    bin_edges_plot = np.linspace(0, 200, 101)
    # bin_edges_plot = np.logspace(0, 1.2, 10)
    plot_hist_2(bin_edges_plot, speed_sm_day, "day", speed_sm_night, "night", "speed mm/s", 1)
    plt.savefig(os.path.join(rootdir, "{0}_hist_D_vs_N_spd_mms.png".format(FISH_ID)))

    # split data into day and night
    position_night_x = np.empty([0, 0])
    position_night_y = np.empty([0, 0])

    position_night_x = np.append(position_night_x, x_nt[np.where(change_times_s[0] > tv_24h_sec)])
    position_night_x = np.append(position_night_x, x_nt[np.where(tv_24h_sec[0:-1] > change_times_s[3])])

    position_night_y = np.append(position_night_y, y_nt[np.where(change_times_s[0] > tv_24h_sec)])
    position_night_y = np.append(position_night_y, y_nt[np.where(tv_24h_sec[0:-1] > change_times_s[3])])

    position_day_x = np.empty([0, 0])
    position_day_y = np.empty([0, 0])

    position_day_x = np.append(position_day_x, x_nt[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                                             (tv_24h_sec[0:-1] < change_times_s[3]))])
    position_day_y = np.append(position_day_y, y_nt[np.where((change_times_s[0] < tv_24h_sec[0:-1]) &
                                                             (tv_24h_sec[0:-1] < change_times_s[3]))])

    # plot position (fix = remove x,y when they were over threshold)
    bin_edges_plot = np.linspace(0, 800, 101)
    plot_hist_2(bin_edges_plot, position_day_y, "day", position_night_y, "night", "Y position", 1)

    xmin = np.nanmin(x_nt[:])
    xmax = np.nanmax(x_nt[:])
    ymin = np.nanmin(y_nt[:])
    ymax = np.nanmax(y_nt[:])
    image_minmax(rootdir, ymin, ymax, FISH_ID, meta)

    fig, (ax1, ax2) = plt.subplots(2, 7, sharey=True)
    for day in range(NUM_DAYS):
        position_night_x = x_nt[np.where((tv_sec > (change_times_s[3] + day_s * day)) &
                                         (tv_sec < (change_times_s[0] + day_s * (day + 1))))]
        position_night_y = y_nt[np.where((tv_sec > (change_times_s[3] + day_s * day)) &
                                         (tv_sec < (change_times_s[0] + day_s * (day + 1))))]

        position_day_x = x_nt[np.where((tv_sec > (change_times_s[0] + day_s * day)) &
                                       (tv_sec < (change_times_s[3] + day_s * day)))]
        position_day_y = y_nt[np.where((tv_sec > (change_times_s[0] + day_s * day)) &
                                       (tv_sec < (change_times_s[3] + day_s * day)))]

        ax1[day].hist2d(position_day_x[~np.isnan(position_day_x)],
                        neg_values(position_day_y[~np.isnan(position_day_y)]),
                        bins=10, range=[[xmin, xmax], [-ymax, ymin]], cmap='inferno')
        ax2[day].hist2d(position_night_x[~np.isnan(position_night_x)],
                        neg_values(position_night_y[~np.isnan(position_night_y)]),
                        bins=10, range=[[xmin, xmax], [-ymax, ymin]], cmap='inferno')
    plt.savefig(os.path.join(rootdir, "{0}_hist2d_D_vs_N_split_days_spt.png".format(FISH_ID)))

    # Distribute position into categories: y [top, centre, bottom], or x [centre side].
    # Assume the fish explores the whole area over the video
    y_bins = 10
    x_bins = 5
    y_bin_size = (ymax - ymin) / y_bins
    x_bin_size = (xmax - xmin) / x_bins

    vertical_pos = np.empty([y_nt.shape[0]])
    previous_y_bin = 0
    for y_bin in range(1, y_bins + 1):
        vertical_pos[np.where(
            np.logical_and((y_nt - ymin) >= previous_y_bin * y_bin_size, (y_nt - ymin) <= y_bin * y_bin_size))] = y_bin
        previous_y_bin = copy.copy(y_bin)

    horizontal_pos = np.empty([x_nt.shape[0]])
    previous_x_bin = 0
    for x_bin in range(1, x_bins + 1):
        horizontal_pos[np.where(
            np.logical_and((x_nt - xmin) >= previous_x_bin * x_bin_size, (x_nt - xmin) <= x_bin * x_bin_size))] = x_bin
        previous_x_bin = copy.copy(x_bin)

    # Bin thresholded data (10fps = seconds, 60 seconds = min e.g. 10*60*10 = 10min bins
    movement = (speed_sm_mm_ps > MOVE_THRESH) * 1
    super_threshold_indices_bin = smooth_speed(movement, 10 * 60 * MIN_BINS)

    # filled plot in s
    plt.close('all')
    fig1, ax1 = filled_plot(tv / 10 ** 9 / 60 / 60, super_threshold_indices_bin, change_times_h,
                            day_ns / 10 ** 9 / 60 / 60)
    ax1.set_ylim([0, 1])
    sec_axis_h(ax1, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Fraction active in {} min sliding windows".format(MIN_BINS))
    plt.title("Fraction_active_{}_thresh_{}_mmps".format(meta["species"], MOVE_THRESH))
    plt.savefig(os.path.join(rootdir, "{0}_wake_spt.png".format(FISH_ID)))

    # win_size = fps * sec/min * mins (was 30*60)
    smooth_win = 10 * 60 * MIN_BINS
    speed_sm_bin = smooth_speed(speed_sm_tbl_ps, win_size=smooth_win)
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed body lengths/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))
    plt.savefig(os.path.join(rootdir, "{0}_speed_30m_spt.png".format(FISH_ID)))

    # win_size = fps * sec/min * mins (was 30*60)
    smooth_win = 10 * 60 * MIN_BINS
    speed_sm_mm_bin = smooth_speed(speed_sm_mm_ps, win_size=smooth_win)
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_mm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed mm/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))
    plt.savefig(os.path.join(rootdir, "{0}_speed_30m_spt.png".format(FISH_ID)))

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_mm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed mm/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))
    ax2.set_ylim(0, 60)
    plt.savefig(
        os.path.join(rootdir, "{0}_speed_30m_spt_0-60ylim.png".format(FISH_ID)))

    smooth_win = 10 * 60 * 10
    speed_sm_mm_bin = smooth_speed(speed_sm_mm_ps, win_size=smooth_win)
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_sm_mm_bin, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax2, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed mm/s")
    plt.title("Speed_{0}_smoothed_by_{1}".format(meta["species"], MIN_BINS))
    plt.savefig(
        os.path.join(rootdir, "{0}_speed_{1}m_spt.png".format(FISH_ID, 10)))

    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, speed_full, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.plot((tv / 10 ** 9 / 60 / 60)[0:-1], speed_t)
    sec_axis_h(ax2, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Speed pixels/0.1s")
    plt.title("Speed_{0}_raw-black_thresholded-blue".format(FISH_ID))
    plt.savefig(os.path.join(rootdir, "{0}_speed_speed_full_speed_thresholded.png".format(FISH_ID)))

    # area
    plt.close()
    fig2, ax2 = filled_plot(tv / 10 ** 9 / 60 / 60, track_full[0:-1, 3], change_times_h, day_ns / 10 ** 9 / 60 / 60)
    plt.xlabel("Time (h)")
    plt.ylabel("Area pixels/0.1s")
    plt.title("Area_{0}".format(meta["species"]))
    sec_axis_h(ax2, video_start_total_sec)
    plt.savefig(os.path.join(rootdir, "{0}_area.png".format(FISH_ID)))

    plt.close()
    fig3, ax3 = filled_plot(tv / 10 ** 9 / 60 / 60, np.diff(tv) / 10 ** 9, change_times_h, day_ns / 10 ** 9 / 60 / 60)
    sec_axis_h(ax3, video_start_total_sec)
    plt.xlabel("Time (h)")
    plt.ylabel("Inter frame time difference (s)")
    plt.title("TV_{0}".format(meta["species"]))
    plt.savefig(os.path.join(rootdir, "{0}_TV_diff.png".format(FISH_ID)))

    # save out track file
    # track file needs: FISH20200727_c1_r1_Genus-species_sex_mmpp_fishlength-mm
    # speed_sm_tbl_ps, tv, x, y, movement
    # speed_sm_mm_ps, tv, x, y

    track_meta = {'ID': FISH_ID, 'species': meta["species"], 'sex': meta["sex"],
                  'fish_length_mm': meta["fish_length_mm"], 'mm_per_pixel': config["mm_per_pixel"]}
    meta_df = pd.DataFrame(track_meta, columns=['ID', 'species', 'sex', 'fish_length_mm', 'mm_per_pixel'], index=[0])
    meta_df.to_csv(os.path.join(rootdir, "{0}_meta.csv".format(FISH_ID)))

    # need to double check this!
    # start from midnight (so they all start at the same time) - need to adjust "midnight" depending on if ts were
    # adjusted for 30min shift (all recordings before 20201127).
    if int(FISH_ID[4:12]) < 20201127:
        thirty_min_ns = 30 * 60 * 1000000000
        adjusted_day_ns = day_ns - thirty_min_ns
        print("old recording from before 20201127 so adjusting back time before saving out als")
    else:
        adjusted_day_ns = day_ns

    midnight = np.max(np.where(tv < adjusted_day_ns))

    track_als = np.vstack((tv[midnight:-1], speed_sm_mm_ps[midnight:, 0], x_nt[midnight:-1], y_nt[midnight:-1]))

    filename = os.path.join(rootdir, "{}_als.csv".format(FISH_ID))
    als_df = pd.DataFrame(track_als.T, columns=['tv_ns', 'speed_mm', 'x_nt', 'y_nt'],
                          index=pd.Index(np.arange(0, len(speed_sm_tbl_ps[midnight:]))))
    als_df.to_csv(filename, encoding='utf-8-sig', index=False)
    plt.close('all')

    # test if saving file worked (issues with null bytes)
    try:
        data_b = pd.read_csv(filename, sep=',')
        # check if all data is as expected
        if data_b.shape != als_df.shape:
            # try  to save again using np
            np.savetxt(filename, track_als.T, delimiter=',', header='tv_ns,speed_mm,x_nt,y_nt', comments='')
            data_b = pd.read_csv(filename, sep=',')
            if data_b.shape != als_df.shape:
                raise Exception("Saving didn't work properly as the saved csv is too short! Report this bug!")
            else:
                print("could save as np")
    except pd.errors.ParserError:
        print("problem parsing, probably null bytes error, trying to save with numpy instead ")
        np.savetxt(filename, track_als.T, delimiter=',', header='tv_ns,speed_mm,x_nt,y_nt', comments='')
        data_b = pd.read_csv(filename, sep=',')
        if data_b.shape != als_df.shape:
            raise Exception("Saving didn't work properly as the saved csv is too short! Report this bug!")
        else:
            print("could save as np")

    try:
        data_b = pd.read_csv(filename, sep=',')
    except pd.errors.ParserError:
        print("still couldn't save it properly, report this!")
        os.remove(filename)
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
