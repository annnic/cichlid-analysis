import glob
import os
import datetime as dt

import pandas as pd

from cichlidanalysis.analysis.processing import remove_cols
from cichlidanalysis.io.tracks import adjust_old_time


def load_als_files(folder, suffix="*als.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    data = []
    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',', error_bad_lines=False, warn_bad_lines=True)
            print("loaded file {}".format(file))
            data_s['FishID'] = file[0:-8]
            data_s['ts'] = adjust_old_time(file, pd.to_datetime(data_s['tv_ns'], unit='ns'))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            # also add in the species name and ID at the start
            data = pd.read_csv(os.path.join(folder, file), sep=',', error_bad_lines=False, warn_bad_lines=True)
            print("loaded file {}".format(file))
            data['FishID'] = file[0:-8]
            data['ts'] = adjust_old_time(file, pd.to_datetime(data['tv_ns'], unit='ns'))
            first_done = 1

    if isinstance(data, pd.DataFrame):
        # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
        data.drop(data.filter(regex="Unname"), axis=1, inplace=True)
        data = data.drop(data[data.ts < dt.datetime.strptime("1970-1-2 00:00:00", '%Y-%m-%d %H:%M:%S')].index)
        data = data.reset_index(drop=True)
        data = remove_cols(data, ['vertical_pos', 'horizontal_pos', 'speed_bl', 'activity'])

    print("All als.csv files loaded")
    return data


def load_bin_als_files(folder, suffix="*als.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

    print("All binned als.csv files loaded")
    return data


def add_condition(data, file, tag1, tag2):
    if tag1 in file:
        data['condition'] = tag1
    elif tag2 in file:
        data['condition'] = tag2
    else:
        data['condition'] = float("nan")
    return data


def load_bin_als_file_condition(folder, suffix="*als_30m.csv", tag1='melatonin', tag2='ethanol'):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data_s = add_condition(data_s, file, tag1, tag2)
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            data = add_condition(data, file, tag1, tag2)
            print("loaded file {}".format(file))
            first_done = 1

    # workaround to deal with Removed index_col=0, as is giving Type error ufunc "isnan'
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

    print("All binned als.csv files loaded")
    return data
