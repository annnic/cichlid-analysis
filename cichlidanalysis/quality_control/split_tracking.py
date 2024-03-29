# In some cases the camera is bumped, the water refilled or something else happens to disturb th video tracking.
# To get around this we use this script. There are two cases which will be taken care of:
# if you want to NaN a region, or if you want to spilt the video and recalculate a background image for each part and
# retrack. This second option also retracks the next video as it assumes that the background used wasn't good enough.

import os
import glob
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import copy
import sys
import datetime

import cv2.cv2 as cv2
import numpy as np

from cichlidanalysis.io.meta import load_yaml, extract_meta
from cichlidanalysis.io.tracks import load_track, get_latest_tracks
from cichlidanalysis.tracking.offline_tracker import tracker
from cichlidanalysis.quality_control.divide_tracking import divide_video
from cichlidanalysis.quality_control.video_tools import background_vid_split


def getFrame(frame_nr):
    global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)


def split_select(video_path, background_cropped):
    """ Function that takes a video path, a median file, and rois. It then uses background subtraction and centroid
    tracking to find the XZ coordinates of the largest contour. This script has a threshold bar which allows you to try
    different levels. Once desired threshold level is found. Press 'q' to quit and the selected value will be used """
    split_start, split_end = [], []
    # load video
    global video
    video = cv2.VideoCapture(video_path)
    nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # set up image display and trackbar for
    cv2.namedWindow('Splitting finder')
    cv2.createTrackbar("Frame", "Splitting finder", 0, nr_of_frames, getFrame)
    cv2.startWindowThread()

    playing = 1
    ret, frame = video.read()
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape != background_cropped.shape:
        # add padding to the median
        if frame_bw.shape[0] != background_cropped.shape[0] and frame_bw.shape[1] == background_cropped.shape[1]:
            background_cropped = np.concatenate((background_cropped, frame_bw[background_cropped.shape[0]:
                                                                              frame_bw.shape[0], :]), axis=0)

        if frame_bw.shape[0] == background_cropped.shape[0] and frame_bw.shape[1] != background_cropped.shape[1]:
            background_cropped = np.concatenate((background_cropped, frame_bw[:, background_cropped.shape[1]:frame_bw.
                                                 shape[1]]), axis=1)

    while video.isOpened():
        if playing:
            ret, frame = video.read()
            if ret:
                frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frameDelta = cv2.absdiff(frame, background_cropped)
                cv2.putText(frameDelta, "Select the start and end of the section to make NaNs", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frameDelta, "Press enter to select frame, press space bar to pause", (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frameDelta, "'a' for backwards, 'd' advance, 'enter' = save out the values", (5, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frameDelta, "frame = {}".format(frame_nr), (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                            (200, 200, 200), 1)
                cv2.putText(frameDelta, "split_start ('s') = {}, split_end ('e') = {}".format(split_start, split_end),
                            (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Splitting finder", frameDelta)

        k = cv2.waitKey(33)

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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elif k == ord("d"):
            frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elif k == ord("s"):
            split_start = video.get(cv2.CAP_PROP_POS_FRAMES)

        elif k == ord("e"):
            split_end = video.get(cv2.CAP_PROP_POS_FRAMES)

        elif k == 13:  # or finished:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            video.release()
            print("Splitting video between frame {} and frame {}".format(split_start, split_end))
            return int(split_start), int(split_end)

    # print("Finished cleaning up")
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # video.release()
    # return int(split_start), int(split_end)


if __name__ == '__main__':
    # find file path for video and load track
    # Allows a user to select file
    root = Tk()
    root.withdraw()
    root.update()
    video_path = askopenfilename(title="Select movie file", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    root.destroy()

    vid_folder_path = os.path.split(video_path)[0]
    vid_timestamp = os.path.split(video_path)[1][0:-10]
    cam_folder_path = os.path.split(vid_folder_path)[0]
    vid_folder_name = os.path.split(vid_folder_path)[1]
    video_name = os.path.split(video_path)[1]

    # find current track path
    track_path = video_path[0:-4] + ".csv"
    if not os.path.isfile(track_path):
        track_path = []
        # movie has been retracked, so pick the right csv
        _, all_files = get_latest_tracks(vid_folder_path, video_path[-9:-4])
        for file in all_files:
            if ("Range" not in file) & (video_name[0:-4] in file):
                track_path = os.path.join(vid_folder_path, file)
            elif ("Range" in file) & (video_name[0:-4] in file):
                print("Careful, ignoring the spilt csv {}".format(track_path))
        print("Using retracked csv {}".format(track_path))
    if not track_path:
        print("Can't find right track! - add issue to github")

    displacement_internal, track_single = load_track(track_path)
    meta = load_yaml(vid_folder_path, "meta_data")
    rois = load_yaml(cam_folder_path, "roi_file")
    new_roi = load_yaml(vid_folder_path, "roi_file")
    config = load_yaml(cam_folder_path, "config")
    fish_data = extract_meta(vid_folder_name)

    os.chdir(cam_folder_path)
    files = glob.glob("*.png")
    files.sort()
    files.insert(0, files.pop(files.index(min(files, key=len))))
    if "{}_Median.png".format(vid_timestamp) in files:
        previous_median_name = files[files.index("{}_Median.png".format(vid_timestamp)) - 1]
        print(previous_median_name)
    else:
        previous_median_name = files[files.index("{}_per90_Background.png".format(vid_timestamp)) - 1]
        print(previous_median_name)

    # find and load background file
    background_path = os.path.join(cam_folder_path, "{}".format(previous_median_name))
    if len(glob.glob(background_path)) != 1:
        print('too many or too few background files in folder:' + cam_folder_path)
        sys.exit()
    else:
        background_full = cv2.imread(glob.glob(background_path)[0], 0)

    roi_n = rois["roi_" + fish_data['roi'][1]]
    background = background_full[roi_n[1]:roi_n[1] + roi_n[3], roi_n[0]:roi_n[0] + roi_n[2]]

    split_s, split_e = split_select(video_path, background)

    while split_e < split_s:
        print("Split start must be smaller than split end, retry")
        split_s, split_e = split_select(video_path, background)

    retrack = 'm'
    while retrack not in {'y', 'n'}:
        retrack = input("Retrack the split movie? y/n: \n")

    track_next_movie = 'm'
    while track_next_movie not in {'y', 'n'}:
        track_next_movie = input("Retrack the movie after the spilt movie? y/n: \n")

    if retrack == 'y':
        # remake backgrounds from the split.
        os.chdir(vid_folder_path)

        # load original track (need timestamps)
        na, track_single_orig = load_track(track_path)

        split_range = ([0, split_s], [split_e, track_single_orig.shape[0]])
        backgrounds = []
        for part in split_range:
            backgrounds.append(background_vid_split(video_path, 100, 90, part))

        # for cases where there wasn't a background made for the post split, don't track second movie and add NaNs
        # until end of movie
        if isinstance(backgrounds[1], list):
            # is empty = extend NaN to end
            fill_zeros_to = split_range[1][1]
        else:
            # There will be a second section tracked so add for middle untracked section NaNs to first track
            fill_zeros_to = split_range[0][1]

        # retrack part of the movie with the correct background. Will also need to use the second background for the
        # movie afterwards making roi for full video
        if new_roi:
            # get the new roi coordinates
            track_rois = {'roi_0': new_roi['roi_0']}
            print("using the new roi")
        else:
            # get the old roi coordinates and reset for the video (so start x,y = 0,0
            width_trim, height_trim = rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
            track_rois = {'roi_0': (0, 0, width_trim, height_trim)}

        for idx, curr_background in enumerate(backgrounds):
            if isinstance(curr_background, np.ndarray):
                #  will skip empty backgrounds
                area_s = 100
                thresh = 35
                tracker(video_path, curr_background, track_rois, threshold=thresh, display=False, area_size=area_s,
                        split_range=split_range[idx])

                # add in the right timepoints (of a primary track - not a full retrack)
                # load the newly tracked csv
                date = datetime.datetime.now().strftime("%Y%m%d")
                range_s = str(split_range[idx][0]).zfill(5)
                range_e = str(split_range[idx][1]).zfill(5)
                filename = video_path[0:-4] + "_tracks_{}_Thresh_{}_Area_{}_Range{}-{}_.csv".format(date, thresh,
                                                                                                    area_s, range_s,
                                                                                                    range_e)
                _, track_single_split = load_track(filename)
                # replace the frame col with the data from the original track
                if idx == 0:
                    # add NaNs (-1 for exclude) and timestamps
                    dummy = np.empty([fill_zeros_to - split_range[0][1], 4])
                    dummy[:] = -1
                    track_single_split = np.concatenate((track_single_split, dummy))
                    track_single_split[:, 0] = track_single_orig[split_range[0][0]:fill_zeros_to, 0]

                else:
                    track_single_split[0:split_range[idx][1] - split_range[idx][0], 0] = track_single_orig[
                                                                                         split_range[idx][0]:
                                                                                         split_range[idx][1], 0]
                # save over
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.savetxt(filename, track_single_split, delimiter=",")

        if track_next_movie == 'y':
            # track the next video (as median will be messed up if it's a background problem.
            # find next movie path
            next_movie_num = str(int(video_name.split("_")[1]) + 1)

            if glob.glob("*_0_roi*.mp4"):
                # old file format
                find_next_movie = glob.glob("*_{}_roi*.mp4".format(next_movie_num))
            else:
                # zfill to 3 digits file format
                find_next_movie = glob.glob("*_{}_roi*.mp4".format(next_movie_num.zfill(3)))

            if find_next_movie:
                next_movie_name = find_next_movie[0]

                if fill_zeros_to == split_range[1][1]:
                    print("Didn't retrack second movie because the second background wasn't made as it was too short. "
                          "In this case, running divide_tracking.py 60min on the next movie")
                    video_path = os.path.join(vid_folder_path, next_movie_name)
                    divide_video(video_path, '60')
                else:
                    # movie has been retracked, so pick the right csv
                    files_c, all_files = get_latest_tracks(vid_folder_path, next_movie_name[:-4])

                    if len(all_files) > 1:
                        print("not retracking {} as there's already multiple track types.".format(next_movie_name))
                    else:
                        next_movie_path = os.path.join(vid_folder_path, next_movie_name)
                        tracker(next_movie_path, backgrounds[1], track_rois, threshold=thresh, display=False, area_size=area_s)

                        # find newly made csv and rename it so it will be used by the loading script.
                        date = datetime.datetime.now().strftime("%Y%m%d")
                        _, new_csv_name = get_latest_tracks(vid_folder_path, next_movie_name[0:-4] + '*' + date)
                        # os.rename(os.path.join(vid_folder_path, new_csv_name), next_movie_path[0:-4] + "_cleaned.csv")

                        # load old and new csv file and replace timestamps
                        _, track_single_retracked = load_track(new_csv_name[0])
                        _, track_single_orig2 = load_track(all_files[0])
                        track_single_retracked[:, 0] = track_single_orig2[:, 0]

                        # add exclude tag to the former csv
                        os.rename(os.path.join(vid_folder_path, all_files[0]),
                                  os.path.join(vid_folder_path, all_files[0][0:-4] + "_exclude.csv"))

                        # # save over
                        resave_path = os.path.join(vid_folder_path, new_csv_name[0])
                        os.makedirs(os.path.dirname(resave_path), exist_ok=True)
                        np.savetxt(resave_path, track_single_retracked, delimiter=",")
            else:
                print("didn't track next movie as couldn't find it (last movie?)")


    else:
        # load track and mark the excluded part (keeps copy of the track as it saves out a "cleaned" version which is
        # prioritised), later '-1' are replaced by nans (after nans from non tracking are interpolated)
        na, track_single = load_track(track_path)
        track_single_cleaned = copy.copy(track_single)
        track_single_cleaned[split_s:split_e, 1:4] = -1

        filename = video_path[0:-4] + "_cleaned.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, track_single_cleaned, delimiter=",")
        print("done")
