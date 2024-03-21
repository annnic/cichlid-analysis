### this script loads a video and it's corresponding track, it plots the centroid over the video and allows you to scroll
# through and control the playback speed
# Inspiration from:
# https://stackoverflow.com/questions/54674343/how-do-we-create-a-trackbar-in-open-cv-so-that-i-can-use-it-to-skip-to-specific
# and for speed graph:
# https://stackoverflow.com/questions/32111705/overlay-a-graph-over-a-video

from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
import sys
import glob

import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cichlidanalysis.io.meta import load_yaml, extract_meta
from cichlidanalysis.io.tracks import extract_tracks_from_fld, get_file_paths_from_nums
from cichlidanalysis.analysis.processing import interpolate_nan_streches, remove_high_spd_xy, smooth_speed, threshold_data
from cichlidanalysis.analysis.behavioural_state import define_rest


def tracker_checker_inputs_grab(video_path_i):
    """ Get the inputs for the tracker_checker

    :param video_path_i:
    :return:
    """
    # ### Movement moving/not-moving use 15mm/s threshold ####
    MOVE_THRESH = 15

    # ### Behavioural state - calculated from Movement ###
    TIME_WINDOW_SEC = 60
    FRACTION_THRESH = 0.05

    FPS = 10

    vid_folder_path = os.path.split(video_path_i)[0]
    vid_timestamp = os.path.split(video_path_i)[1][0:-10]
    cam_folder_path = os.path.split(vid_folder_path)[0]
    vid_folder = os.path.split(vid_folder_path)[1]

    track_single_i, displacement_internal = extract_tracks_from_fld(vid_folder_path, vid_timestamp)
    vid_name = os.path.split(video_path_i)[1]

    meta = load_yaml(vid_folder_path, "meta_data")
    config = load_yaml(cam_folder_path, "config")
    fish_data = extract_meta(vid_folder)

    new_rois = load_yaml(vid_folder_path, "roi_file")
    rois = load_yaml(cam_folder_path, "roi_file")

    os.chdir(cam_folder_path)
    roi_n = rois["roi_" + fish_data['roi'][1]]

    # interpolate between NaN streches
    try:
        x_n = interpolate_nan_streches(track_single_i[:, 1])
        y_n = interpolate_nan_streches(track_single_i[:, 2])
    except:
        x_n = track_single_i[:, 1]
        y_n = track_single_i[:, 2]

    if new_rois:
        # subtract the difference so that the centroids are plotted at the right coordinates
        # output: (x,y,w,h)
        # assume that if there is a new ROI, there is only one.
        x_n += new_rois['roi_{}'.format('0')][0]
        y_n += new_rois['roi_{}'.format('0')][1]
        track_single_i[:, 1] += new_rois['roi_{}'.format('0')][0]
        track_single_i[:, 2] += new_rois['roi_{}'.format('0')][1]
        roi_n = new_rois['roi_{}'.format('0')]

        # add in ROI to video
        start_point = (roi_n[0], roi_n[1])
        end_point = (roi_n[0] + roi_n[2], roi_n[1] + roi_n[3])
    else:
        start_point = (0, 0)
        end_point = (roi_n[2], roi_n[3])

    # find displacement
    displacement_i_mm_s = displacement_internal * config["mm_per_pixel"] * config['fps']
    speed_full_i = np.sqrt(np.diff(x_n) ** 2 + np.diff(y_n) ** 2)
    speed_t, x_nt_i, y_nt_i = remove_high_spd_xy(speed_full_i, x_n, y_n)

    spd_sm = smooth_speed(speed_t, win_size=5)
    spd_sm_mm = spd_sm * config["mm_per_pixel"]
    spd_sm_mm_ps = spd_sm_mm * config['fps']

    # threshold the speed_mm with 15mm/s
    movement = threshold_data(spd_sm_mm_ps, MOVE_THRESH)
    move_df = pd.DataFrame({'movement': movement})
    rest = ((move_df.loc[:, 'movement'].transform(lambda s: s.rolling(FPS * TIME_WINDOW_SEC).mean())) < FRACTION_THRESH) * 1
    rest = rest.to_frame(name='rest')

    return spd_sm, spd_sm_mm_ps, MOVE_THRESH, rest, move_df, displacement_i_mm_s, vid_name, track_single_i, start_point, \
           end_point, x_nt_i, y_nt_i, config, vid_folder


# # function called by trackbar, sets the next frame to be read
def get_frame(frame_nr, video):
    # global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)


def track_checker_gui(video_path_j, spd_sm, spd_sm_mm_ps, thresh, rest, move_df, displacement_i_mm_s,
                      vid_name, track_single_i, start_point, end_point, x_nt, y_nt, config, video_folder_path, vid_folder):
    """ this script loads a video and it's corresponding track, it plots the centroid over the video and allows you to
    scroll through the video. Prints the ROI if it is in the video folder (indidcating a new ROI).

    :param video_path_j:

    :param spd_sm:
    :param spd_sm_mm_ps:
    :param thresh:
    :param displacement_i_mm_s:
    :param vid_name:
    :param track_single_i:
    :param start_point:
    :param end_point:
    :param x_nt:
    :param y_nt:
    :return:
    """
    # open video
    video = cv2.VideoCapture(video_path_j)

    # get total number of frames
    nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # create display window
    cv2.namedWindow("Speed of {}".format(vid_name))

    # add track bar
    cv2.createTrackbar("Frame", "Speed of {}".format(vid_name), 0, nr_of_frames, lambda f: get_frame(f, video))

    ret, frame = video.read()
    # height, width = frame.shape[:2]
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    max_sp = np.nanmax(spd_sm_mm_ps)
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    playing = 1

    while 1:
        # show frame, break the loop if no frame is found
        curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES) - 1

        if ret:
            # try:
            #     cX, cY = (int(track_single_i[int(curr_frame), 1]), int(track_single_i[int(curr_frame), 2]))
            #     cv2.circle(frame, (int(x_nt[int(curr_frame)]), int(y_nt[int(curr_frame)])), 4, (0, 255, 255), 4)
            #     cv2.circle(frame, (cX, cY), 4, (0, 0, 255), 2)
            #     cv2.putText(frame, "yellow: corrected centroid, red: raw centroid, frame: {}".format(curr_frame),
            #                 (5, 15),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 200), 1)
            #     cv2.rectangle(frame, start_point, end_point, 220, 2)
            # except:
            scale_bar_pixels = int(100/config["mm_per_pixel"])
            cv2.rectangle(img=frame, pt1=(50, 10), pt2=(scale_bar_pixels+50, 20), color=(0, 0, 255), thickness=-1)

            cv2.putText(frame, "10cm", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 200), 1)

            cv2.imshow("Speed of {}".format(vid_name), frame)

        ax.clear()
        dummy = np.zeros([int(max_sp), 400, 3]).astype(np.uint8)
        plt.imshow(dummy, origin='lower')

        if curr_frame > 200:
            track_curr = 200
        else:
            track_curr = curr_frame
        plt.plot([track_curr, track_curr], [-0.01, max_sp], color='r')

        win_min = int(curr_frame - 20 * 10)
        win_max = int(curr_frame + 20 * 10)

        if win_min < 0:
            win_min = 0
        if win_max > spd_sm.shape[0]:
            win_max = spd_sm.shape[0]

        values = spd_sm_mm_ps[win_min:win_max]
        values_2 = displacement_i_mm_s[win_min:win_max]
        rest_vals = rest.loc[win_min:win_max].values
        move_vals = move_df.loc[win_min:win_max].values
        plt.plot(values_2)
        plt.plot(values, linewidth=3)
        plt.plot(5 + rest_vals * thresh, linewidth=3)
        plt.plot(7 + move_vals * thresh, linewidth=3)
        # plt.plot([0, 400], [thresh, thresh])
        plt.plot([0, 400], [15, 15])

        ax.set_ylim([0, max_sp])
        ax.set_xlim([0, 400])
        ax.set_aspect('auto')
        # ax.legend(["current frame", "speed_sm_mm_ps", "thresh 15mm/s"])
        ax.legend(["current frame", "speed_raw_mm_ps", "speed_sm_mm_ps", "rest y:1, n:0", "move y:1, n:0", "threshold {}".format(thresh)])
        plt.ylabel("mm/s")

        k = cv2.waitKey(33)
        # stop playback when q is pressed
        if k == 27 or k == ord("q"):
            cv2.destroyAllWindows()
            break

        elif k == 32:
            if playing == 0:
                playing = 1
            elif playing == 1:
                playing = 0

        elif k == ord("a"):
            frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr - 2)
            ret, frame = video.read()

        elif k == ord("d"):
            frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
            ret, frame = video.read()

        elif k == ord("n"):
            # release resources
            plt.close()
            video.release()
            cv2.destroyAllWindows()
            return True

        elif k == ord("p"):
            cv2.putText(frame, "frame#: {}".format(curr_frame), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, vid_folder, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 200), 1)

            # Save the frame
            cv2.imwrite(os.path.join(video_folder_path,
                                     "snapshot_{}_movie_Date{}_frame_{}.png".format(vid_folder, vid_name[0:-4],
                                                                                    int(curr_frame))), frame)

        if playing:
            # Get the next videoframe
            ret, frame = video.read()
        else:
            ret = True

    # release resources
    video.release()
    cv2.destroyAllWindows()
    return False


def run_tracker_checker_grab():
    """ Based off For running the tracker checker. Allows you to define the movie number and folder

    :return:
    """
    # define movie number to check
    video_num = '-1'
    while int(video_num) == -1:
        video_num = input("What is the number of the movie you would like to check?:")

    # Allows a user to select folder
    root = Tk()
    video_folder_path = askdirectory(parent=root, title="Select movie file")
    root.destroy()

    video_path = get_file_paths_from_nums(video_folder_path, video_num, file_format='*.mp4')
    if len(video_path):
        video_path = video_path[0]
    else:
        print("Couldn't find the movie. Exiting")
        return

    next_vid = True

    while next_vid is True:
        speed_sm, speed_sm_mm_ps, threshold, rest, move_df, displacement_internal_mm_s, video_name, \
        track_single, s_point, e_point, x_nt, y_nt, config, vid_folder = tracker_checker_inputs_grab(video_path)

        next_vid = track_checker_gui(video_path, speed_sm, speed_sm_mm_ps, threshold, rest, move_df,
                                     displacement_internal_mm_s,
                                     video_name, track_single, s_point, e_point, x_nt, y_nt, config, video_folder_path,
                                     vid_folder)
        next_movie_n = "_" + str(int(video_path.split('_')[-2]) + 1).zfill(len(video_path.split('_')[-2])) + "_"

        os.chdir(os.path.split(video_path)[0])
        video_files = glob.glob("*.mp4")
        for vid in video_files:
            if next_movie_n in vid:
                video_path = os.path.join(os.path.split(video_path)[0], vid)


if __name__ == '__main__':
    run_tracker_checker_grab()
