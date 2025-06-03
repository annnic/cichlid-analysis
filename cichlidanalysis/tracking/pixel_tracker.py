import datetime
import os
import time
import glob

from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import Tk
import cv2.cv2 as cv2
import numpy as np

from cichlidanalysis.tracking.rois import define_roi_still
from cichlidanalysis.tracking.helpers_pixels import correct_tags_pixel
from cichlidanalysis.tracking.backgrounds import background_vid, update_background
from cichlidanalysis.io.meta import extract_meta, load_yaml
from cichlidanalysis.io.tracks import remove_tags, get_file_paths_from_nums
from cichlidanalysis.io.movies import get_movie_paths


def pixel_delta(video_path, background_full, rois, threshold=5, display=True, area_size=0, split_range=False):
    """ Function that takes a video path, a background file, rois, threshold and display switch. This then uses
    background subtraction and centroid tracking to find the XZ coordinates of the largest contour. Saves out a csv file
     with frame #, X, Y, contour area"""

    print("tracking {}".format(video_path))

    # As camera is often excluded, check here and buffer if not included
    if len(rois) == 1:
        rois['cam'] = 'unknown'

    # load video
    video = cv2.VideoCapture(video_path)

    if display:
        # create display window
        cv2.namedWindow("Live thresholded")
        cv2.namedWindow("Live")

    # as there can be multiple rois the writer, data and moviename are kept in lists
    data = list()
    frame_id = 0
    for roi in np.arange(0, len(rois) - 1):
        data.append(list())

    firstframe = True

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("reached end of video")
            video.release()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameDelta_full = cv2.absdiff(background_full, gray)
        image_thresholded = cv2.threshold(frameDelta_full, threshold, 255, cv2.THRESH_TOZERO)[1]

        for roi in range(0, len(rois) - 1):
            # for the frame define an ROI and crop image
            curr_roi = rois["roi_" + str(roi)]
            image_thresholded_roi = image_thresholded[curr_roi[1]:curr_roi[1] + curr_roi[3],
                         curr_roi[0]:curr_roi[0] + curr_roi[2]]
            if firstframe:
                # skip as can't subtract
                firstframe = False

            else:
                last_image_thresholded_roi = last_image_thresholded[curr_roi[1]:curr_roi[1] + curr_roi[3],
                                             curr_roi[0]:curr_roi[0] + curr_roi[2]]

                delta_pixels = cv2.absdiff(last_image_thresholded_roi, image_thresholded_roi)
                delta_pixels_logic = delta_pixels > 0
                data[roi].append((frame_id, delta_pixels_logic.sum()))

            # cv2.imwrite('{}_per{}_background.png'.format('test', 90), image_thresholded)
        last_image_thresholded = image_thresholded

        if frame_id % 500 == 0:
            print("Frame {}".format(frame_id))

        if display:
            full_image_thresholded = (cv2.threshold(frameDelta_full, threshold, 255, cv2.THRESH_TOZERO)[1])
            # Live display of full resolution and ROIs
            cv2.putText(full_image_thresholded, "Framenum: {}".format(frame_id), (30,
                                                                                  full_image_thresholded.shape[0] -
                                                                                  30), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=255)

            cv2.imshow("Live thresholded", full_image_thresholded)
            cv2.imshow("Live", gray)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    # saving data
    print("Saving data output")
    date = datetime.datetime.now().strftime("%Y%m%d")

    for roi in range(0, len(rois) - 1):
        datanp = np.array(data[roi])
        filename = video_path[0:-4] + "_delta-pixel_{}_Thresh_{}_roi-{}.csv".format(date, threshold,
                                                                                           roi)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            # os.path.isdir(os.path.dirname(filename))
            np.savetxt(filename, datanp, delimiter=",")
        except:
            print("issue with saving,trying again")
            time.sleep(2)
            os.path.isdir(os.path.dirname(filename))
            try:
                np.savetxt(filename, datanp, delimiter=",")
            except:
                print('trying to save after 2s failed')

    print("Tracking finished on video cleaning up")
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    background_update = 'm'
    while background_update not in {'y', 'n'}:
        background_update = input("Update background? y/n: \n")

    if background_update == 'y':
        background_update_files = 'm'
        while background_update_files not in {'a', 'n'}:
            background_update_files = input("Update background for all movies (a) or one (n)?: \n")

        percentile = 90
        if background_update_files == 'a':
            # percentile = input("Run with which percentile? 90 is default")
            update_background(percentile)

        elif background_update_files == 'n':
            root = Tk()
            root.withdraw()
            root.update()
            video_file_back = askopenfilename(title="Select movie file",
                                              filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            root.destroy()

            # percentile = input("Run with which percentile? 90 is default")
            background_vid(video_file_back, 200, percentile)

    # track_videos = 'm'
    # while track_videos not in {'y', 'n'}:
    #     track_videos = input("Track videos? y/n: \n")
    #

    track_videos = 'm'
    while track_videos not in {'y', 'n'}:
        track_videos = input("Track videos? y/n: \n")

    if track_videos == 'y':
        track_all = 'm'
        while track_all not in {'y', 'n', 's'}:
            track_all = input("Track all videos (y)? one video (n) or select videos (s): \n")

        if track_all == 'y':
            # Allows a user to select top directory
            root = Tk()
            root.withdraw()
            root.update()
            vid_dir = askdirectory()
            root.destroy()

            os.chdir(vid_dir)
            video_files = glob.glob("*.mp4")
            video_files.sort()

            cam_dir = os.path.split(vid_dir)[0]

            backgrounds = glob.glob("*background.png")
            backgrounds = remove_tags(backgrounds, remove=["frame"])
            new_bgd = True
            if len(backgrounds) < 1:
                print("Didn't find remade background, will use original background in camera folder")
                os.chdir(cam_dir)
                backgrounds = glob.glob("*ackground.png")
                new_bgd = False
            backgrounds.sort()

            rec_name = os.path.split(vid_dir)[1]

        elif track_all == 's':
            print("only works for remade backgrounds")
            video_paths, vid_dir, video_nums = get_movie_paths()
            video_files = []
            for i in video_paths:
                video_files.append(os.path.split(i)[1])
            backgrounds = get_file_paths_from_nums(vid_dir, video_nums, file_format='*.png')
            cam_dir = os.path.split(vid_dir)[0]
            new_bgd = True
            if backgrounds == []:
                print("couldn't  find backgrounds, looking at old backgrounds)")
                new_bgd = False
                backgrounds = get_file_paths_from_nums(cam_dir, video_nums, file_format='*.png')
            rec_name = os.path.split(vid_dir)[1]

        elif track_all == 'n':
            # Allows a user to select top directory
            root = Tk()
            root.withdraw()
            root.update()
            video_file = askopenfilename(title="Select movie file",
                                         filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            root.destroy()

            vid_dir = os.path.split(video_file)[0]
            cam_dir = os.path.split(vid_dir)[0]
            rec_name = os.path.split(vid_dir)[1]
            video_files = [os.path.split(video_file)[1]]

            os.chdir(os.path.split(video_file)[0])
            backgrounds = glob.glob(video_file[0:-4] + "*background.png")
            backgrounds = remove_tags(backgrounds, remove=["frame"])
            new_bgd = True
            if len(backgrounds) < 1:
                print("Didn't find remade background, will use original background in camera folder")
                os.chdir(cam_dir)
                backgrounds = glob.glob("*" + video_file[-20:-10] + "*.png")
                new_bgd = False

        fish_data = extract_meta(rec_name)

        track_roi = 'm'
        while track_roi not in {'y', 'n'}:
            track_roi = input("Track with another roi? y/n: \n")

        if track_roi == 'y':
            if track_all == 'n':
                roi_on_one = input("You are now changing the ROI for  only one video, this  is not recommended!\n "
                                   "y to continue, n to  stop: \n")
                if roi_on_one == 'n':
                    exit()

            # ##  Define video rois ##
            # load recording roi
            rec_rois = load_yaml(cam_dir, "roi_file")
            curr_roi = rec_rois["roi_" + str(fish_data['roi'][1:])]

            # load video roi (if previously defined) or if not, then pick background and define a new ROI
            vid_rois = load_yaml(vid_dir, "roi_file")
            if not vid_rois:
                # allow user to pick the background image which to set the roi with
                root = Tk()
                root.withdraw()
                root.update()
                background_file = askopenfilename(title="Select background", filetypes=(("image files", "*.png"),))
                root.destroy()
                background_full = cv2.imread(background_file)

                # crop background to roi
                # have issue where roi can go over and then cropping the right background is an issue, rare case
                if curr_roi[1] + curr_roi[3] > background_full.shape[0]:
                    print("something off with roi/background size... readjusting")
                    off_by_on_y = background_full.shape[0] - (curr_roi[1] + curr_roi[3])
                    curr_roi = (curr_roi[0], curr_roi[1] + off_by_on_y, curr_roi[2], curr_roi[3])

                background_crop = background_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                     curr_roi[2]]
                if background_crop.ndim == 3:
                    background_crop = cv2.cvtColor(background_crop, cv2.COLOR_BGR2GRAY)

                define_roi_still(background_crop, vid_dir)
                vid_rois = load_yaml(vid_dir, "roi_file")

            for idx, val in enumerate(video_files):
                movie_n = val.split("_")[1]
                background_of_movie = [i for i in backgrounds if i.split("_")[1] == movie_n]
                if not background_of_movie:
                    print("didn't find background, stopping tracking")
                    break
                print("tracking with background {}".format(background_of_movie))
                background_full = cv2.imread(background_of_movie[0], 0)
                if new_bgd:
                    background_crop = background_full
                else:
                    background_crop = background_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                         curr_roi[2]]
                pixel_delta(os.path.join(vid_dir, val), background_crop, vid_rois, threshold=35, display=False,
                        area_size=100)

        else:
            vid_rois = load_yaml(cam_dir, "roi_file")
            width_trim, height_trim = vid_rois['roi_{}'.format(fish_data['roi'][-1])][2:4]
            rois = {'roi_0': (0, 0, width_trim, height_trim)}

            for idx, val in enumerate(video_files):
                movie_n = val.split("_")[1]
                background_of_movie = [i for i in backgrounds if (i.split('/')[-1]).split("_")[1] == movie_n]
                print("tracking with background {}".format(background_of_movie[0]))
                background = cv2.imread(background_of_movie[0], 0)

                # check if using an old background (need to crop) or new
                if new_bgd:
                    background_crop = background
                else:
                    curr_roi_n = vid_dir.split("_")[-3][1]
                    curr_roi = vid_rois['roi_{}'.format(curr_roi_n)]
                    background_crop = background[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] +
                                                                                                 curr_roi[2]]
                    # extremely rarely the background needs to be padded, this hack can be used
                    # Used for:
                    # FISH20211103_c5_r1_Lepidiolamprologus-elongatus_su, FISH20211006_c3_r0_Neolamprologus-brevis_su
                    # import numpy as np
                    # background_crop = np.vstack([background_crop, np.zeros([1, curr_roi[2]], dtype='uint8')])

                pixel_delta(os.path.join(vid_dir, val), background_crop, rois, threshold=35, display=False, area_size=100)

        # find cases where a movie has multiple csv files, add exclude tag to the ones from not today (date in file
        # names) and replace timestamps.
        date = datetime.datetime.now().strftime("%Y%m%d")
        correct_tags_pixel(date, vid_dir)
