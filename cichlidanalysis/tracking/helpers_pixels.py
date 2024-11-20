import os
import glob
import copy

import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import Tk

from cichlidanalysis.io.tracks import load_track, remove_tags
from cichlidanalysis.tracking.helpers import exclude_tag_csv


def copy_timestamp_pixels(orig_csv_path_i, new_csv_path_i):
    """This script will copy the timestamps from a original track"""
    # load csv file and replace timestamps
    _, track_single_orig = load_track(orig_csv_path_i)
    track_single_retracked = np.genfromtxt(new_csv_path_i, delimiter=',')

    if track_single_retracked[0, 0] == 1:
        track_single_retracked[:, 0] = track_single_orig[1:, 0]

        # save over
        os.makedirs(os.path.dirname(new_csv_path_i), exist_ok=True)
        np.savetxt(new_csv_path_i, track_single_retracked, delimiter=",")
    else:
        print("Timestamps already copied")
    return


def update_csvs_pixels(orig_csv_path, new_csv_path):
    """This script will copy the timestamps from a original track, and rename the old track so it has the "exclude" tag.
    This will not work on "range" files"""
    orig_csv_name = os.path.split(orig_csv_path)[1]
    # orig_csv_folder = os.path.split(orig_csv_path)[0]
    new_csv_name = os.path.split(new_csv_path)[1]

    # check that file is not a range file
    if "Range" in new_csv_name:
        print("new csv file is a range, not adding exclude tag")
    elif "exclude" in orig_csv_name:
        copy_timestamp_pixels(orig_csv_path, new_csv_path)
        print("old csv file already has exclude tag, copying timestamps but not adding tag")
    else:
        copy_timestamp_pixels(orig_csv_path, new_csv_path)
        # add "exclude" tag to old csv track file
        exclude_tag_csv(orig_csv_path)
    return


def correct_tags_pixel(retracking_date, vid_directory):
    """ find cases where a movie has multiple csv files, add exclude tag to the ones from not today (date in file
    names), exclude any old tracks, check if timestamps have been replaced.

    :param retracking_date: "%Y%m%d"
    :param vid_directory: path of a recording
    :return:
    """

    os.chdir(vid_directory)
    video_files = glob.glob("*.mp4")
    video_files.sort()

    # find all csvs
    all_files = glob.glob("*.csv")
    all_files.sort()
    all_files = remove_tags(all_files, remove=["meta.csv", "als.csv"])

    # find csvs with retracking date (which are the recent ones)
    new_files = glob.glob("*_{}_*.csv".format(retracking_date))
    new_files.sort()

    if len(new_files) == 0:
        print("no new csv files found with this date - double check date")
        return

    old_files = [file_a for file_a in all_files if file_a not in new_files]

    # ## find split tracks and replace with excluded old track ##
    splits = []
    for track in old_files:
        if "_Range" in track:
            splits.append(track)

    # find original tracks for each movie as these have the timestamps
    orig_tracks = copy.copy(old_files)
    orig_tracks = remove_tags(orig_tracks, remove=["meta.csv", "als.csv", "_Range", "cleaned"])
    orig_tracks.sort()

    # add old timestamps to newly re-tracked data
    for orig_n, orig_file in enumerate(orig_tracks):
        for new_f in new_files:
            if orig_file[0:18] == new_f[0:18]:
                print("updating timestamps of {} and adding exclude tag to {}".format(new_f, orig_file))
                update_csvs_pixels(os.path.join(vid_directory, orig_file), os.path.join(vid_directory, new_f))

    old_files_excluding_orig = [file_a for file_a in old_files if file_a not in orig_tracks]

    # add exclude tag to the old files (excluding orig - which were already "exlcuded" in last lines:
    for file in old_files_excluding_orig:
        exclude_tag_csv(os.path.join(vid_directory, file))
    return


if __name__ == '__main__':

    correct_new_tracks = 'm'
    while correct_new_tracks not in {'y', 'n'}:
        correct_new_tracks = input("Correct new tracks? y/n: \n")

    if correct_new_tracks == 'y':
        date_tag = input("What is the date tag for the new tracks? YYYYMMDD: \n")
        # Allows a user to select top directory
        root = Tk()
        rootdir = askdirectory(parent=root, title="Select video folder (which has the movies and tracks)")
        root.destroy()

        correct_tags_pixel(date_tag, rootdir)
