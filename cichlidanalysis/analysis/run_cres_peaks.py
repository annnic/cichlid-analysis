import warnings
import os

import datetime as dt

import pandas as pd

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.utils.timings import load_timings, load_timings_14_8
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks_min, crespuscular_weekly_fish, \
    crespuscular_daily_ave_fish, crepuscular_peaks_min_daily
from cichlidanalysis.analysis.run_binned_als import setup_run_binned
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box, \
    plot_day_night_species_ave, plot_cre_dawn_dusk_stacked, plot_cre_dawn_dusk_peak_loc, \
    plot_cre_dawn_dusk_peak_loc_bin_size, plot_cre_dawn_dusk_peak_loc_bin_size_by_diel_guild, plot_cre_dawn_dusk_loc_strip_box
from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg


if __name__ == '__main__':
    rootdir = select_dir_path()

    bin_size_min = 10

    fish_tracks_bin, sp_metrics, _, species_full, fish_IDs, species_sixes = setup_run_binned(rootdir, als_type='*als_{}m.csv'.format(bin_size_min))
    loadings = pd.read_csv(os.path.join(rootdir, 'pca_loadings.csv'))

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit \
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    feature = 'speed_mm'

    all_peaks_df = crepuscular_peaks_min_daily(rootdir, feature, fish_tracks_bin, change_times_m,
                                               bin_size_min=bin_size_min, peak_prom=7)
    peak_loc_mean_dawn = all_peaks_df.loc[all_peaks_df.twilight == 'dawn'].groupby('species').mean()
    peak_loc_mean_dusk = all_peaks_df.loc[all_peaks_df.twilight == 'dusk'].groupby('species').mean()

    # find how many times there's a peak location and only include species which have at least 50% of indiviudals
    # which have a peak
    period = 'dawn'
    cres_peaks_subset = all_peaks_df.loc[all_peaks_df.twilight == period]
    df_counts = cres_peaks_subset.groupby('species').count()
    peak_fraction = df_counts.peak_loc / df_counts.FishID
    peak_fraction_dawn = peak_fraction.to_frame(name='peak_fraction')

    period = 'dusk'
    cres_peaks_subset = all_peaks_df.loc[all_peaks_df.twilight == period]
    df_counts = cres_peaks_subset.groupby('species').count()
    peak_fraction = df_counts.peak_loc / df_counts.FishID
    peak_fraction_dusk = peak_fraction.to_frame(name='peak_fraction')

    # correlations
    data1 = peak_fraction_dawn.peak_fraction
    data2 = peak_fraction_dusk.peak_fraction
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq, name_x='dawn', name_y='dusk', labels=False)

    model, r_sq = run_linear_reg(data1, loadings.set_index('species').pc1)
    plt_lin_reg(rootdir, data1, loadings.set_index('species').pc1, model, r_sq, name_x='dawn', name_y='pc1')
    model, r_sq = run_linear_reg(data1, loadings.set_index('species').pc2)
    plt_lin_reg(rootdir, data1, loadings.set_index('species').pc2, model, r_sq, name_x='dawn', name_y='pc2')
    model, r_sq = run_linear_reg(data2, loadings.set_index('species').pc1)
    plt_lin_reg(rootdir, data2, loadings.set_index('species').pc1, model, r_sq, name_x='dusk', name_y='pc1')
    model, r_sq = run_linear_reg(data2, loadings.set_index('species').pc2)
    plt_lin_reg(rootdir, data2, loadings.set_index('species').pc2, model, r_sq, name_x='dusk', name_y='pc2')

    # peak location correlations, need to drop nans
    to_drop_dawn = peak_loc_mean_dawn.loc[peak_loc_mean_dawn.isna().values].index
    data3 = peak_loc_mean_dawn.drop(to_drop_dawn)
    data4 = loadings.set_index('species').drop(to_drop_dawn)
    model, r_sq = run_linear_reg(data3.peak_loc, data4.pc1)
    plt_lin_reg(rootdir, data3.peak_loc, data4.pc1, model, r_sq, name_x='dawn peak loc', name_y='pc1')
    model, r_sq = run_linear_reg(data3.peak_loc, data4.pc2)
    plt_lin_reg(rootdir, data3.peak_loc, data4.pc2, model, r_sq, name_x='dawn peak loc', name_y='pc2')


    to_drop_dusk = peak_loc_mean_dusk.loc[peak_loc_mean_dusk.isna().values].index
    data5 = peak_loc_mean_dusk.drop(to_drop_dusk)
    data6 = loadings.set_index('species').drop(to_drop_dusk)
    model, r_sq = run_linear_reg(data5.peak_loc, data6.pc1)
    plt_lin_reg(rootdir, data5.peak_loc, data6.pc1, model, r_sq, name_x='dusk peak loc', name_y='pc1')
    model, r_sq = run_linear_reg(data5.peak_loc, data6.pc2)
    plt_lin_reg(rootdir, data5.peak_loc, data6.pc2, model, r_sq, name_x='dusk peak loc', name_y='pc2')

    plot_cre_dawn_dusk_loc_strip_box(rootdir, all_peaks_df, feature, peak_feature='peak_loc', bin_size_min=30)

    diel_guilds = pd.read_csv(os.path.join(rootdir, 'diel_guilds.csv'), sep=',')
    plot_cre_dawn_dusk_peak_loc_bin_size_by_diel_guild(rootdir, all_peaks_df, feature, change_times_m, 'average', diel_guilds,
                                                       peak_feature='peak_loc', bin_size_min=bin_size_min)

    plot_cre_dawn_dusk_peak_loc_bin_size(rootdir, all_peaks_df, feature, change_times_m, name='average',
                                         peak_feature='peak_loc',
                                         bin_size_min=bin_size_min)

    # # for plotting peaks of an individual species
    crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_bin, change_times_m, bin_size_min=bin_size_min)
    # crespuscular_weekly_fish(rootdir, feature, fish_tracks_bin, change_times_m, bin_size_min=bin_size_min)

    # plot_cre_dawn_dusk_peak_loc(rootdir, all_peaks_df, feature, change_times_unit, name='average', peak_feature='peak_loc')



    # # get better look at the timing of peaks?
    # plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature, peak_feature='peak_amplitude')
    # plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature, peak_feature='peak')
    # plot_cre_dawn_dusk_peak_loc(rootdir, cres_peaks, feature, change_times_unit, name='average', peak_feature='peak_loc')
    # plot_cre_dawn_dusk_peak_loc(rootdir, cres_peaks_indiv, feature, change_times_unit, name='individual',
    #                             peak_feature='peak_loc')
    # plot_cre_dawn_dusk_stacked(rootdir, cres_peaks, feature, peak_feature='peak')


