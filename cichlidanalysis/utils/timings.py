import math
import numpy as np
import datetime as dt
import os
import glob


def infer_tv(fps, filechunk):
    """Takes fps and the length of the video to compute an inferred time vector that starts from 0
    >>> infer_tv(1, 3)
    array([0., 1., 2.])
    """
    tv = np.arange(0, filechunk, 1 / fps)
    return tv


def output_timings():
    """ set sunrise, day, sunset, night times (ns, s, m, h).
    Sunrise starts at 7am, Day 7.30am, Sunset 6.30pm, Night 7pm

    :return:
    """
    change_times_s = [7*60*60, 7*60*60 + 30*60, 18*60*60 + 30*60, 19*60*60]
    change_times_ns = [i * 10**9 for i in change_times_s]
    change_times_m = [i / 60 for i in change_times_s]
    change_times_h = [i / 60 / 60 for i in change_times_s]
    change_times_d = [i / 24 for i in change_times_h]
    change_times_unit = [7 * 2, 7.5 * 2, 18.5 * 2, 19 * 2]

    # set day in ns
    day_ns = 24 * 60 * 60 * 10**9
    day_s = 24 * 60 * 60

    change_times_datetime = [dt.datetime.strptime("1970-1-2 07:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 07:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 18:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 19:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S')]

    return change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, \
           change_times_datetime, change_times_unit


def output_timings_14_8():
    """ set sunrise, day, sunset, night times (ns, s, m, h).
    Sunrise starts at 7am, Day 7.30am, Sunset 6.30pm, Night 7pm

    :return:
    """
    change_times_s = [8*60*60, 8*60*60 + 30*60, 21*60*60 + 30*60, 22*60*60]
    change_times_ns = [i * 10**9 for i in change_times_s]
    change_times_m = [i / 60 for i in change_times_s]
    change_times_h = [i / 60 / 60 for i in change_times_s]
    change_times_d = [i / 24 for i in change_times_h]
    change_times_unit = [8 * 2, 8.5 * 2, 21.5 * 2, 22 * 2]

    # set day in ns
    day_ns = 24 * 60 * 60 * 10**9
    day_s = 24 * 60 * 60

    change_times_datetime = [dt.datetime.strptime("1970-1-2 08:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 08:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 21:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 22:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S')]

    return change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, \
           change_times_datetime, change_times_unit


def output_timings_730():
    """ set sunrise, day, sunset, night times (ns, s, m, h). For old data where:
    Sunrise starts at 7.30am, Day 8.00am, Sunset 7.00pm, Night 7.30pm

    :return:
    """
    change_times_s = [7*60*60 + 30*60, 8*60*60, 19*60*60, 19*60*60 + 30*60]
    change_times_ns = [i * 10**9 for i in change_times_s]
    change_times_m = [i / 60 for i in change_times_s]
    change_times_h = [i / 60 / 60 for i in change_times_s]
    change_times_d = [i / 24 for i in change_times_h]

    # set day in ns
    day_ns = 24 * 60 * 60 * 10**9
    day_s = 24 * 60 * 60
    return change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d


def load_timings(vector_length):
    """ Get all of the required time parameters. All data is with fps = 10 (may change in future

    :param vector_length:
    :return:
    """
    fps = 10

    # get time variables
    change_times_s, change_times_ns, change_times_m, change_times_h, day_ns, day_s, change_times_d, \
    change_times_datetime, change_times_unit = output_timings()
    tv_ns = np.arange(0, day_ns * 8, 10 ** 9 / 10)
    tv_ns = tv_ns[0:vector_length]
    tv_sec = tv_ns / 10 ** 9

    # correct to 24h time
    tv_24h_sec = tv_ns / 10 ** 9
    num_days = 7
    tv_s_type = get_time_state(tv_sec, day_s, change_times_s, fps)

    return fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
           day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit


def get_time_state(tv_i_s, day_unit_s, change_times_unit_s, fps):
    """ Input must be in seconds!!! state of day, night = 0, dawn/dusk = 1, daylight = 2
    """

    days_to_plot = math.ceil(tv_i_s[-1]/day_unit_s)
    time_state = np.zeros([len(tv_i_s), 1])
    change_times_unit_fps = [element * fps for element in change_times_unit_s]
    day_unit = day_unit_s*fps

    for day_n in range(days_to_plot):
        # night times
        time_state[0+day_unit * day_n:change_times_unit_fps[0] + day_unit * day_n] = 0
        time_state[change_times_unit_fps[3]+day_unit * day_n:day_unit + day_unit * day_n] = 0

        # dawn
        time_state[change_times_unit_fps[0] + day_unit * day_n:change_times_unit_fps[1]+day_unit * day_n] = 1

        # day
        time_state[change_times_unit_fps[1] + day_unit * day_n:change_times_unit_fps[2]+day_unit * day_n] = 2

        # dusk
        time_state[change_times_unit_fps[2] + day_unit * day_n:change_times_unit_fps[3]+day_unit * day_n] = 1

    return time_state


def get_start_time_from_str(start_time):
    """ From a HHMMSS string get the start time in seconds"""
    start_total_sec = (int(start_time[0:2]) * 60 * 60 + int(start_time[2:4]) * 60 + int(start_time[4:]))
    return start_total_sec


def get_start_time_of_video(rootdir):
    os.chdir(rootdir)
    files = glob.glob("*.csv")
    files.sort()
    start_time = files[0][9:15]
    start_total_sec = get_start_time_from_str(start_time)
    return start_total_sec


def set_time_vector(track_full, video_start_total_sec, config):
    """ Get time vector from the data, or if there isn't data in that column then interpolate the tv"""
    NS_IN_SECONDS = 10 ** 9

    if track_full[0, 0] == 0:
        tv = np.arange(video_start_total_sec * NS_IN_SECONDS,
                       ((track_full.shape[0] / config['fps']) * NS_IN_SECONDS + video_start_total_sec * NS_IN_SECONDS),
                       ((1 / config['fps']) * NS_IN_SECONDS))
        print("using retracked data so using interpolated time vector")
    else:
        tv = track_full[:, 0] - track_full[0, 0] + video_start_total_sec * NS_IN_SECONDS
    return tv


def load_timings_14_8(vector_length):
    """ Get all of the required time parameters. All data is with fps = 10 (may change in future

    :param vector_length:
    :return:
    """
    fps = 10

    # get time variables
    change_times_s_14_8, change_times_ns_14_8, change_times_m_14_8, change_times_h_14_8, day_ns, day_s, \
    change_times_d_14_8, change_times_datetime_14_8, change_times_unit_14_8 = output_timings_14_8()
    tv_ns = np.arange(0, day_ns * 8, 10 ** 9 / 10)
    tv_ns = tv_ns[0:vector_length]
    tv_sec = tv_ns / 10 ** 9

    # correct to 24h time
    tv_24h_sec = tv_ns / 10 ** 9
    num_days = 7
    tv_s_type = get_time_state(tv_sec, day_s, change_times_s_14_8, fps)

    return fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s_14_8, change_times_ns_14_8, \
           change_times_h_14_8, day_ns, day_s, change_times_d_14_8, change_times_m_14_8, change_times_datetime_14_8,\
           change_times_unit_14_8


if __name__ == "__main__":
    import doctest
    doctest.testmod()
