# This script holds functions for making a background image of a video

import datetime

import numpy as np
import cv2.cv2 as cv2
import os
import glob
from tkinter.filedialog import askdirectory
from tkinter import Tk


def background_vid(videofilepath, nth_frame, percentile):
    """ (str, int, int, int)
     This function will create a background image of the defined area"""
    try:
        cap = cv2.VideoCapture(videofilepath)
    except:
        print("problem reading video file, check path")
        return

    counter = 0
    gatheredFramess = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        if counter % nth_frame == 0:
            print("Frame {}".format(counter))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # settings with Blur settings for the video loop
            gatheredFramess.append(image)
        counter += 1

    background = np.percentile(gatheredFramess, int(percentile), axis=0).astype(dtype=np.uint8)
    cv2.imshow('Calculated Background from {} percentile'.format(percentile), background)

    # background = np.percentile(frameMedian, 90, axis=0).astype(dtype=np.uint8)
    # cv2.imshow('Calculated background', background)

    date = datetime.datetime.now().strftime("%Y%m%d")
    vid_name = videofilepath[0:-4]
    # vid_name = os.path.split(videofilepath)[1][0:-4]
    # vid_folder_path = os.path.split(videofilepath)[0]
    cv2.imwrite('{}_per{}_background.png'.format(vid_name, percentile), background)

    cap.release()
    cv2.destroyAllWindows()
    return background


def update_background(percentile, vid_type="*.mp4"):
    # Allows a user to select top directory
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory()
    root.destroy()

    os.chdir(rootdir)
    files = glob.glob(vid_type)
    files.sort()

    for video in files:
        background_vid(os.path.join(rootdir, video), 200, percentile)
