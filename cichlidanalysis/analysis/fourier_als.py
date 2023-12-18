# inspiration from https://realpython.com/python-scipy-fft/
import os

from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt


def run_rfft_fishtracks(rootdir, fish_tracks):
    """ Runs fast fourier transform on real value input. Need the ts in %Y-%m-%d %H:%M:%S format (as str or in dt
    format) and "speed_mm" values. Computes the sample rate from the difference between ts[0] and ts[1].

    :param rootdir:
    :param fish_tracks_bin:
    :return:
    """
    for fish in fish_tracks.FishID.unique():
        fish_track_bin = fish_tracks.loc[fish_tracks.FishID == fish, ['ts', 'speed_mm']]

    if isinstance(fish_track_bin.iloc[0, 0], str):
        first_ts = dt.datetime.strptime(fish_track_bin.iloc[0, 0], '%Y-%m-%d %H:%M:%S')
        second_ts = dt.datetime.strptime(fish_track_bin.iloc[1, 0], '%Y-%m-%d %H:%M:%S')
    else:
        first_ts = fish_track_bin.iloc[0, 0]
        second_ts = fish_track_bin.iloc[1, 0]

    SAMPLE_RATE_HZ = 1/(second_ts - first_ts).total_seconds()
    # DURATION_DT = dt.datetime.strptime(fish_track_bin.iloc[-1, 0], '%Y-%m-%d %H:%M:%S') - \
    #              dt.datetime.strptime(fish_track_bin.iloc[0, 0], '%Y-%m-%d %H:%M:%S')
    # DURATION_S = DURATION_DT.total_seconds()
    # N = int(SAMPLE_RATE_HZ * DURATION_S)

    num_elements = len(fish_track_bin.speed_mm.to_numpy())

    yf = rfft(fish_track_bin.speed_mm.to_numpy())
    xf = rfftfreq(num_elements, 1 / SAMPLE_RATE_HZ)

    plt.axvline(1/(60*60*24), c='gainsboro')
    plt.plot(xf, np.abs(yf))
    plt.savefig(os.path.join(rootdir, "rfft_{}.png".format(fish)))
    plt.close()
    return
