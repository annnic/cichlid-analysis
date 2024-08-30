##########
# This software is licensed under the GNU General Public License v3.0
# Copyright (c) 2019 Vassilis Bitsikas
# Modified by Annika Nichols
##########

import datetime
import os
import time

import cv2.cv2 as cv2
import numpy as np


def tracker(video_path, background_full, rois, threshold=5, display=True, area_size=0, split_range=False):
    """ Function that takes a video path, a background file, rois, threshold and display switch. This then uses
    background subtraction and thresholding to find pixel changes. Saves out a csv file with pixel changes"""

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

    if split_range is False:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        split_range = [0, total + 1]
        split_name = False
    else:
        split_name = True

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("reached end of video")
            video.release()
            break
        if frame_id in np.arange(split_range[0], split_range[1]):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameDelta_full = cv2.absdiff(background_full, gray)

            # tracking
            cx = list()
            cy = list()
            contourOI = list()
            contourOI_ = list()
            for roi in range(0, len(rois) - 1):
                # for the frame define an ROI and crop image
                curr_roi = rois["roi_" + str(roi)]
                frameDelta = frameDelta_full[curr_roi[1]:curr_roi[1] + curr_roi[3],
                             curr_roi[0]:curr_roi[0] + curr_roi[2]]
                image_thresholded = cv2.threshold(frameDelta, threshold, 255, cv2.THRESH_TOZERO)[1]
                (contours, _) = cv2.findContours(image_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    contourOI_.append(max(contours, key=cv2.contourArea))
                    area = cv2.contourArea(contourOI_[roi])
                    if area > area_size:
                        contourOI.append(cv2.convexHull(contourOI_[roi]))
                        M = cv2.moments(contourOI[roi])
                        cx.append(int(M["m10"] / M["m00"]))
                        cy.append(int(M["m01"] / M["m00"]))
                        data[roi].append((frame_id, cx[roi], cy[roi], area))
                    else:
                        print("no large enough contour found for roi {}!".format(roi))
                        data[roi].append((frame_id, np.nan, np.nan, np.nan))
                        contourOI_[-1] = False
                        contourOI.append(False)
                        cx.append(np.nan)
                        cy.append(np.nan)
                else:
                    print("no contour found for roi {}!".format(roi))
                    data[roi].append((frame_id, np.nan, np.nan, np.nan))
                    contourOI_.append(False)
                    contourOI.append(False)
                    cx.append(np.nan)
                    cy.append(np.nan)

            if frame_id % 500 == 0:
                print("Frame {}".format(frame_id))
            if display:
                full_image_thresholded = (cv2.threshold(frameDelta_full, threshold, 255, cv2.THRESH_TOZERO)[1])
                # Live display of full resolution and ROIs
                cv2.putText(full_image_thresholded, "Framenum: {}".format(frame_id), (30,
                                                                                      full_image_thresholded.shape[0] -
                                                                                      30), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=255)

                for roi in range(0, len(rois) - 1):
                    if np.all(contourOI_[roi] != False):
                        curr_roi = rois["roi_" + str(roi)]
                        # add in contours
                        corrected_contour = np.empty(contourOI_[roi].shape)
                        corrected_contour[:, 0, 0] = contourOI_[roi][:, 0, 0] + curr_roi[0]
                        corrected_contour[:, 0, 1] = contourOI_[roi][:, 0, 1] + curr_roi[1]
                        cv2.drawContours(full_image_thresholded, corrected_contour.astype(int), -1, 255, 1)

                        # add in centroid
                        cv2.circle(full_image_thresholded, (cx[roi] + curr_roi[0], cy[roi] + curr_roi[1]), 8, 255, 1)

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
        if split_name is False:
            filename = video_path[0:-4] + "_tracks_{}_Thresh_{}_Area_{}_roi-{}.csv".format(date, threshold, area_size,
                                                                                           roi)
        else:
            range_s = str(split_range[0]).zfill(5)
            range_e = str(split_range[1]).zfill(5)
            filename = video_path[0:-4] + "_tracks_{}_Thresh_{}_Area_{}_Range{}-{}_.csv".format(date, threshold,
                                                                                                area_size, range_s,
                                                                                                range_e)
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
