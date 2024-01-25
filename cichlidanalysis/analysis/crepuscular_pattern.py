import copy
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator)

from cichlidanalysis.analysis.processing import species_feature_fish_daily_ave
from cichlidanalysis.analysis.diel_pattern import replace_crep_peaks, make_fish_peaks_df
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_peaks


def crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_ds, species):
    """ Finds peaks in crepuscular periods for the daily average of each fish for each species

    :param rootdir:
    :param feature:
    :param fish_tracks_ds:
    :param species:
    :return:
    """

    border_top = np.ones(48)
    border_bottom = np.ones(48) * 1.05
    dawn_s, dawn_e, dusk_s, dusk_e = [6 * 2, 8 * 2, 18 * 2, 20 * 2]
    border_bottom[6 * 2:8 * 2] = 0
    border_bottom[18 * 2:20 * 2] = 0

    peak_prom = 0.15
    if feature == 'speed_mm':
        border_top = border_top * 200
        border_bottom = border_bottom * 200
        peak_prom = 7

    for species_name in species:
        fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)

        if feature == 'rest':
            fish_daily_ave_feature = np.abs(fish_daily_ave_feature - 1)

        fig1, ax1 = plt.subplots(figsize=(5, 5))
        sns.heatmap(fish_daily_ave_feature.T, cmap="Greys")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.axvspan(dawn_s, dawn_e, color='wheat', alpha=0.5, linewidth=0)
        ax2.axvspan(dusk_s, dusk_e, color='wheat', alpha=0.5, linewidth=0)
        for i in np.arange(0, len(fish_daily_ave_feature.columns)):
            x = fish_daily_ave_feature.iloc[:, i]
            peaks, _ = find_peaks(x, distance=4, prominence=peak_prom, height=(border_bottom, border_top))
            ax2.plot(x)
            ax2.plot(peaks, x[peaks], "o", color="r")
            ax1.plot(x.reset_index().index[peaks].values, (np.ones(len(peaks)) * i) + 0.5, "o", color="r")
            plt.title(species_name)
        plt.savefig(os.path.join(rootdir, "cres_peaks_daily_{}.png".format(species_name)), dpi=1200)
    plt.close('all')


def crespuscular_weekly_fish(rootdir, feature, fish_tracks_ds, species, change_times_m, bin_size_min=30):
    """ Finds peaks in crepuscular periods for the weekly data of each fish for each species

    :param rootdir:
    :param feature:
    :param fish_tracks_ds:
    :param species:
    :return:
    """

    num_day_bins, bins_per_h, _, _, _, _, _, border_bottom_week, border_top_week = peak_borders(bin_size_min, change_times_m)

    peak_prom = 0.15
    if feature == 'speed_mm':
        border_bottom_week = border_bottom_week * 200
        border_top_week = border_top_week * 200
        peak_prom = 7
    dawn_s, dawn_e, dusk_s, dusk_e = (change_times_m[0]-60)/60, (change_times_m[0]+60)/60, (change_times_m[3]-60)/60, (change_times_m[3] + 60) / 60

    date_form = DateFormatter("%H")

    for species_name in fish_tracks_ds.species.unique():
        fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
            columns='FishID', values=feature, index='ts')

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.heatmap(fish_feature.T, cmap="Greys")

        fig2, ax2 = plt.subplots(len(fish_feature.columns), 1, figsize=(6, 2*len(fish_feature.columns)))
        ax2[-1].set_ylabel(feature)
        for fish, fish_name in enumerate(fish_feature.columns):
            for day in range(7):
                ax2[fish].axvspan(dawn_s * bins_per_h + day * num_day_bins, dawn_e * bins_per_h + day * num_day_bins,
                            color='wheat', alpha=0.5, linewidth=0)
                ax2[fish].axvspan(dusk_s * bins_per_h + day * num_day_bins, dusk_e * bins_per_h + day * num_day_bins,
                                  color='wheat', alpha=0.5, linewidth=0)
                ax2[fish].axvspan(0 + day * num_day_bins, dawn_s * bins_per_h + day * num_day_bins,
                            color='lightblue', alpha=0.5, linewidth=0)
                ax2[fish].axvspan(dusk_e * bins_per_h + day * num_day_bins, num_day_bins + day * num_day_bins,
                            color='lightblue', alpha=0.5, linewidth=0)


            x = fish_feature.iloc[:, fish]
            peaks, peak_prop = find_peaks(x, distance=bins_per_h*2, prominence=peak_prom, height=(border_bottom_week[0:x.shape[0]],
                                                                                       border_top_week[0:x.shape[0]]))

            np.around(peak_prop['peak_heights'], 2)

            ax2[fish].plot(x, linewidth= 0.5)
            ax2[fish].plot(peaks, x[peaks], "o", c="r", markersize=3)
            ax2[fish].title.set_text(fish_name)

            ax1.plot(x.reset_index().index[peaks].values, (np.ones(len(peaks)) * fish) + 0.5, "o", color="r")
            ax2[fish].xaxis.set_major_locator(MultipleLocator(num_day_bins))
            ax2[fish].xaxis.set_major_formatter(date_form)
            plt.xlabel("Time (h)")
            plt.ylabel("Speed (mm/s)")
            sns.despine(top=True, right=True)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "cres_peaks_weekly_{}_bin_size_{}min.png".format(species_name, bin_size_min)), dpi=350)
        plt.close()

        ax1.xaxis.set_major_locator(MultipleLocator(24))
        ax1.xaxis.set_major_formatter(date_form)
        plt.xlabel("Time (h)")
        plt.ylabel("Speed (mm/s)")
        sns.despine(top=True, right=True)
        plt.savefig(os.path.join(rootdir, "cres_peaks_weekly_heatmap_{}.png".format(species_name)), dpi=350)
    return



def crepuscular_peaks(rootdir, feature, fish_tracks_ds, fish_diel_patterns_sp):
    """ Uses borders to find peaks in the twilight periods (2h -/+ of sunup/down). Finds peak position and height.
    Uses the defined diel pattern to define baseline, nocturnal = night, diurnal = day, undefined = mean of day and
    night. This version allows you to use to only use 30min bins

    # Returns: peak height, peak location, dawn/dusk, max day/night for that day, if peak missing, find most common
    peak, if all peaks missing use average of the whole period location and use the value of that bin.
    Find amplitude of peaks

    :param feature:
    :param fish_tracks_ds:
    :param fish_diel_patterns: df with: 'FishID', 'peak_amplitude', 'peak', 'twilight', 'species', 'species_six'
    :return:
    """

    # define borders
    border_top = np.ones(48)
    border_bottom = np.ones(48) * 1.05
    dawn_border_bottom = copy.copy(border_bottom)
    dawn_border_bottom[6 * 2:(8 * 2)] = 0
    dusk_border_bottom = copy.copy(border_bottom)
    dusk_border_bottom[18 * 2:(20 * 2)] = 0

    border = np.zeros(48)
    day_border = copy.copy(border)
    day_border[8 * 2:18 * 2] = 1
    night_border = copy.copy(border)
    night_border[0:6 * 2] = 1
    night_border[20 * 2:24 * 2] = 1

    peak_prom = 0.15
    if feature == 'speed_mm':
        border_top = border_top * 200
        dawn_border_bottom = dawn_border_bottom * 200
        dusk_border_bottom = dusk_border_bottom * 200
        peak_prom = 7

    first_all = True
    for species_name in fish_tracks_ds.species.unique():
        fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
            columns='FishID', values=feature, index='ts')
        first = True
        for fish in np.arange(0, len(fish_feature.columns)):
            epoques = np.arange(0, 48 * 7.5, 48).astype(int)

            # create dummies
            fish_peaks_dawn = np.zeros([4, int(np.floor(fish_feature.iloc[:, fish].reset_index().shape[0] / 48))])
            fish_peaks_dusk = np.zeros([4, int(np.floor(fish_feature.iloc[:, fish].reset_index().shape[0] / 48))])

            # for each epoque (48 time points for 30min binned data) find the peaks in dawn and dusk
            for j in np.arange(0, int(np.ceil(fish_feature.shape[0] / 48))):
                x = fish_feature.iloc[epoques[j]:epoques[j + 1], fish]
                if x.size == 48:
                    dawn_peak, dawn_peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=(
                        dawn_border_bottom[0:x.shape[0]], border_top[0:x.shape[0]]))

                    dusk_peak, dusk_peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=(
                        dusk_border_bottom[0:x.shape[0]], border_top[0:x.shape[0]]))

                    # fish_peaks data: position of peak within 24h, position of peak within week, raw peak height,
                    # raw peak height - baseline
                    if dawn_peak.size != 0:
                        fish_peaks_dawn[0, j] = dawn_peak[0]
                        fish_peaks_dawn[1, j] = dawn_peak[0] + epoques[j]
                        fish_peaks_dawn[2, j] = np.round(dawn_peak_prop['peak_heights'][0], 2)

                    if dusk_peak.size != 0:
                        fish_peaks_dusk[0, j] = dusk_peak[0]
                        fish_peaks_dusk[1, j] = dusk_peak[0] + epoques[j]
                        fish_peaks_dusk[2, j] = np.round(dusk_peak_prop['peak_heights'][0], 2)

                    day_mean = np.round(x[day_border.astype(int) == 1].mean(), 2)
                    night_mean = np.round(x[night_border.astype(int) == 1].mean(), 2)
                    daynight_mean = np.round(x[(night_border + day_border).astype(int) == 1].mean(), 2)

                    # how the baseline is chosen is dependent on the diel pattern of the species of the fish
                    # pattern = fish_diel_patterns.loc[fish_diel_patterns.FishID == fish_feature.columns[i],
                    #                                  'species_diel_pattern'].values[0]
                    pattern = fish_diel_patterns_sp.loc[fish_diel_patterns_sp.species == species_name,
                                                        'diel_pattern'].values[0]
                    if pattern == 'nocturnal':
                        fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - night_mean
                        fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - night_mean
                    elif pattern == 'diurnal':
                        fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - day_mean
                        fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - day_mean
                    elif pattern == 'undefined':
                        fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - daynight_mean
                        fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - daynight_mean
                    else:
                        print("pattern not known, stopping function on {}".format(fish_feature.columns[fish]))
                        return

            fish_peaks_dawn = replace_crep_peaks(fish_peaks_dawn, fish_feature, fish, epoques)
            fish_peaks_dusk = replace_crep_peaks(fish_peaks_dusk, fish_feature, fish, epoques)

            # # plotting
            # savedir = '/Users/annikanichols/Desktop/'
            # plot_speed_30m_peaks(savedir, fish_feature.iloc[:, fish], fish_peaks_dawn, fish_peaks_dusk)

            fish_peaks_df_dawn = make_fish_peaks_df(fish_peaks_dawn, fish_feature.columns[fish])
            fish_peaks_df_dusk = make_fish_peaks_df(fish_peaks_dusk, fish_feature.columns[fish])

            fish_peaks_df_dawn['twilight'] = 'dawn'
            fish_peaks_df_dusk['twilight'] = 'dusk'
            fish_peaks_df = pd.concat([fish_peaks_df_dawn, fish_peaks_df_dusk], axis=0)

            if first:
                species_peaks_df = fish_peaks_df
                first = False
            else:
                species_peaks_df = pd.concat([species_peaks_df, fish_peaks_df], axis=0)
        species_peaks_df['species'] = species_name

        if first_all:
            all_peaks_df = species_peaks_df
            first_all = False
        else:
            all_peaks_df = pd.concat([all_peaks_df, species_peaks_df], axis=0)

    all_peaks_df = all_peaks_df.reset_index(drop=True)
    all_peaks_df['peak'] = (all_peaks_df.peak_loc != 0) * 1
    all_peaks_df.loc[all_peaks_df.peak_loc == 0, 'peak_loc'] = np.nan

    # average for each fish for dawn and dusk for 'peak_amplitude', peaks/(peaks+nonpeaks)
    periods = ['dawn', 'dusk']
    first_all = True
    for species_name in fish_tracks_ds.species.unique():
        first = True
        for period in periods:
            feature_i = all_peaks_df[(all_peaks_df['species'] == species_name) & (all_peaks_df['twilight'] == period)][
                ['peak_amplitude', 'FishID', 'crep_num', 'peak', 'peak_loc']]
            sp_average_peak_amp = feature_i.groupby('FishID').mean().peak_amplitude.reset_index()
            sp_average_peak = feature_i.groupby('FishID').mean().peak.reset_index()
            sp_average_peak_loc = feature_i.groupby('FishID').mean().peak_loc.reset_index()
            sp_average_peak_data = pd.concat([sp_average_peak_amp, sp_average_peak, sp_average_peak_loc], axis=1)
            sp_average_peak_data['twilight'] = period
            if first:
                sp_feature_combined = sp_average_peak_data
                first = False
            else:
                sp_feature_combined = pd.concat([sp_feature_combined, sp_average_peak_data], axis=0)
        sp_feature_combined['species'] = species_name

        if first_all:
            all_feature_combined = sp_feature_combined
            first_all = False
        else:
            all_feature_combined = pd.concat([all_feature_combined, sp_feature_combined], axis=0)
    all_feature_combined = all_feature_combined.reset_index(drop=True)
    all_feature_combined = all_feature_combined.loc[:, ~all_feature_combined.columns.duplicated()]

    return all_feature_combined, all_peaks_df


def crepuscular_peaks_min(rootdir, feature, fish_tracks_ds, change_times_m, bin_size_min=30, peak_prom=7):
    """ Uses borders to find peaks in the twilight periods (2h -/+ of sunup/down). Finds peak position and height.
    Uses the defined diel pattern to define baseline, nocturnal = night, diurnal = day, undefined = mean of day and
    night. This version allows you to use different binning (not just 30min)

    # Returns: peak height, peak location, dawn/dusk, max day/night for that day, if peak missing, find most common
    peak, if all peaks missing use average of the whole period location and use the value of that bin.
    Find amplitude of peaks

    :param feature:
    :param fish_tracks_ds:
    :param fish_diel_patterns: df with: 'FishID', 'peak_amplitude', 'peak', 'twilight', 'species', 'species_six'
    :return:
    """

    num_day_bins, bins_per_h, dawn_border_bottom, dusk_border_bottom, border_top, day_border, night_border, _, _ = \
        peak_borders(bin_size_min, change_times_m)

    if feature == 'speed_mm':
        border_top = border_top * 200
        dawn_border_bottom = dawn_border_bottom * 200
        dusk_border_bottom = dusk_border_bottom * 200

    first_all = True
    for species_name in fish_tracks_ds.species.unique():
        fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
            columns='FishID', values=feature, index='ts')
        first = True
        for fish in np.arange(0, len(fish_feature.columns)):
            days = np.arange(0, bins_per_h*24 * 7.5, num_day_bins).astype(int)

            # create dummies
            fish_peaks_dawn = np.zeros([4, int(np.floor(fish_feature.iloc[:, fish].reset_index().shape[0] / num_day_bins))])
            fish_peaks_dusk = np.zeros([4, int(np.floor(fish_feature.iloc[:, fish].reset_index().shape[0] / num_day_bins))])

            # for each epoque (48 time points for 30min binned data) find the peaks in dawn and dusk
            for j in np.arange(0, int(np.ceil(fish_feature.shape[0] / num_day_bins))):
                x = fish_feature.iloc[days[j]:days[j + 1], fish]
                if x.size == num_day_bins:
                    dawn_peak, dawn_peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=(
                        dawn_border_bottom[0:x.shape[0]], border_top[0:x.shape[0]]))
                    plt.savefig(os.path.join(rootdir, "speed.png"), dpi=350)

                    dusk_peak, dusk_peak_prop = find_peaks(x, distance=4, prominence=peak_prom, height=(
                        dusk_border_bottom[0:x.shape[0]], border_top[0:x.shape[0]]))

                    # plt.plot(x)
                    # plt.savefig(os.path.join(rootdir, "speed.png"), dpi=350)

                    # fish_peaks data: position of peak within 24h, position of peak within week, raw peak height,
                    # raw peak height - baseline
                    if dawn_peak.size != 0:
                        fish_peaks_dawn[0, j] = dawn_peak[0]
                        fish_peaks_dawn[1, j] = dawn_peak[0] + days[j]
                        fish_peaks_dawn[2, j] = np.round(dawn_peak_prop['peak_heights'][0], 2)

                    if dusk_peak.size != 0:
                        fish_peaks_dusk[0, j] = dusk_peak[0]
                        fish_peaks_dusk[1, j] = dusk_peak[0] + days[j]
                        fish_peaks_dusk[2, j] = np.round(dusk_peak_prop['peak_heights'][0], 2)

                    day_mean = np.round(x[day_border.astype(int) == 1].mean(), 2)
                    night_mean = np.round(x[night_border.astype(int) == 1].mean(), 2)
                    daynight_mean = np.round(x[(night_border + day_border).astype(int) == 1].mean(), 2)

                    # # how the baseline is chosen is dependent on the diel pattern of the species of the fish
                    # # pattern = fish_diel_patterns.loc[fish_diel_patterns.FishID == fish_feature.columns[i],
                    # #                                  'species_diel_pattern'].values[0]
                    # pattern = fish_diel_patterns_sp.loc[fish_diel_patterns_sp.species == species_name,
                    #                                     'diel_pattern'].values[0]
                    # if pattern == 'nocturnal':
                    #     fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - night_mean
                    #     fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - night_mean
                    # elif pattern == 'diurnal':
                    #     fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - day_mean
                    #     fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - day_mean
                    # elif pattern == 'undefined':
                    #     fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - daynight_mean
                    #     fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - daynight_mean
                    # else:
                    #     print("pattern not known, stopping function on {}".format(fish_feature.columns[fish]))
                    #     return

            fish_peaks_dawn = replace_crep_peaks(fish_peaks_dawn, fish_feature, fish, days)
            fish_peaks_dusk = replace_crep_peaks(fish_peaks_dusk, fish_feature, fish, days)

            # # plotting
            # savedir = '/Users/annikanichols/Desktop/'
            # plot_speed_30m_peaks(savedir, fish_feature.iloc[:, fish], fish_peaks_dawn, fish_peaks_dusk)

            fish_peaks_df_dawn = make_fish_peaks_df(fish_peaks_dawn, fish_feature.columns[fish])
            fish_peaks_df_dusk = make_fish_peaks_df(fish_peaks_dusk, fish_feature.columns[fish])

            fish_peaks_df_dawn['twilight'] = 'dawn'
            fish_peaks_df_dusk['twilight'] = 'dusk'
            fish_peaks_df = pd.concat([fish_peaks_df_dawn, fish_peaks_df_dusk], axis=0)

            if first:
                species_peaks_df = fish_peaks_df
                first = False
            else:
                species_peaks_df = pd.concat([species_peaks_df, fish_peaks_df], axis=0)
        species_peaks_df['species'] = species_name

        if first_all:
            all_peaks_df = species_peaks_df
            first_all = False
        else:
            all_peaks_df = pd.concat([all_peaks_df, species_peaks_df], axis=0)

    all_peaks_df = all_peaks_df.reset_index(drop=True)
    all_peaks_df['peak'] = (all_peaks_df.peak_loc != 0) * 1
    all_peaks_df.loc[all_peaks_df.peak_loc == 0, 'peak_loc'] = np.nan

    # average for each fish for dawn and dusk for 'peak_amplitude', peaks/(peaks+nonpeaks)
    periods = ['dawn', 'dusk']
    first_all = True
    for species_name in fish_tracks_ds.species.unique():
        first = True
        for period in periods:
            feature_i = all_peaks_df[(all_peaks_df['species'] == species_name) & (all_peaks_df['twilight'] == period)][
                ['peak_amplitude', 'FishID', 'crep_num', 'peak', 'peak_loc']]
            sp_average_peak_amp = feature_i.groupby('FishID').mean().peak_amplitude.reset_index()
            sp_average_peak = feature_i.groupby('FishID').mean().peak.reset_index()
            sp_average_peak_loc = feature_i.groupby('FishID').mean().peak_loc.reset_index()
            sp_average_peak_data = pd.concat([sp_average_peak_amp, sp_average_peak, sp_average_peak_loc], axis=1)
            sp_average_peak_data['twilight'] = period
            if first:
                sp_feature_combined = sp_average_peak_data
                first = False
            else:
                sp_feature_combined = pd.concat([sp_feature_combined, sp_average_peak_data], axis=0)
        sp_feature_combined['species'] = species_name

        if first_all:
            all_feature_combined = sp_feature_combined
            first_all = False
        else:
            all_feature_combined = pd.concat([all_feature_combined, sp_feature_combined], axis=0)
    all_feature_combined = all_feature_combined.reset_index(drop=True)
    all_feature_combined = all_feature_combined.loc[:, ~all_feature_combined.columns.duplicated()]

    return all_feature_combined, all_peaks_df


def peak_borders(bin_size_min, change_times_m):
    # define borders - depends on bin size. e.g. 1h bins = 24
    num_day_bins = 24*60/bin_size_min
    bins_per_h = 60/bin_size_min
    if (num_day_bins % 1) != 0:
        print("The 24h day must be divisible by this minute bin size")
        return
    num_day_bins = int(num_day_bins)
    border_top = np.ones(num_day_bins)
    dawn_dusk_border_bottom = np.ones(num_day_bins) * 1.05
    dawn_border_bottom = copy.copy(dawn_dusk_border_bottom)
    dawn_s = int(((change_times_m[0] - 60)/60)*bins_per_h)
    dawn_e = int(((change_times_m[0] + 60)/60)*bins_per_h)
    dawn_border_bottom[dawn_s:dawn_e] = 0

    dusk_border_bottom = copy.copy(dawn_dusk_border_bottom)
    dusk_s = int(((change_times_m[3] - 60)/60)*bins_per_h)
    dusk_e = int(((change_times_m[3] + 60)/60)*bins_per_h)
    dusk_border_bottom[dusk_s:dusk_e] = 0

    dawn_dusk_border_bottom[dawn_s:dawn_e] = 0
    dawn_dusk_border_bottom[dusk_s:dusk_e] = 0

    border_bottom_week = np.concatenate((dawn_dusk_border_bottom, dawn_dusk_border_bottom, dawn_dusk_border_bottom,
                                         dawn_dusk_border_bottom, dawn_dusk_border_bottom, dawn_dusk_border_bottom,
                                         dawn_dusk_border_bottom))
    border_top_week = np.ones(num_day_bins * 7)

    border = np.zeros(num_day_bins)
    day_border = copy.copy(border)
    day_border[int(((change_times_m[0] + 60)/60)*bins_per_h):int(((change_times_m[3] - 60)/60)*bins_per_h)] = 1
    night_border = copy.copy(border)
    night_border[0:int(((change_times_m[0] - 60)/60)*bins_per_h)] = 1
    night_border[int(((change_times_m[3] + 60)/60)*bins_per_h):int(24*bins_per_h)] = 1

    return num_day_bins, bins_per_h, dawn_border_bottom, dusk_border_bottom, border_top, day_border, night_border, border_bottom_week, border_top_week
