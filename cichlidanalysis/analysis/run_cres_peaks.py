import warnings
import os

import datetime as dt

import pandas as pd

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks_min, crespuscular_weekly_fish, \
    crespuscular_daily_ave_fish, crepuscular_peaks_min_daily
from cichlidanalysis.analysis.run_binned_als import setup_run_binned
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box, \
    plot_day_night_species_ave, plot_cre_dawn_dusk_stacked, plot_cre_dawn_dusk_peak_loc, \
    plot_cre_dawn_dusk_peak_loc_bin_size, plot_cre_dawn_dusk_peak_loc_bin_size_by_diel_guild
from cichlidanalysis.io.als_files import load_bin_als_files
from cichlidanalysis.utils.timings import load_timings_14_8


if __name__ == '__main__':
    rootdir = select_dir_path()

    bin_size_min = 10
    # fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, \
    # species_sixes = setup_run_binned(rootdir, als_type='*als_{}m.csv'.format(bin_size_min))
    fish_tracks_bin = load_bin_als_files(rootdir, '*als_{}m.csv'.format(bin_size_min))
    fish_IDs = fish_tracks_bin['FishID'].unique()

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit \
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    feature = 'speed_mm'

    crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_bin, change_times_m, bin_size_min=bin_size_min)
    # crespuscular_weekly_fish(rootdir, feature, fish_tracks_bin, change_times_m, bin_size_min=bin_size_min)

    all_peaks_df = crepuscular_peaks_min_daily(rootdir, feature, fish_tracks_bin, change_times_m,
                                               bin_size_min=bin_size_min, peak_prom=7)
    plot_cre_dawn_dusk_peak_loc_bin_size(rootdir, all_peaks_df, feature, change_times_m, name='average',
                                         peak_feature='peak_loc',
                                         bin_size_min=bin_size_min)
    diel_guilds = ata_s = pd.read_csv(os.path.join(rootdir, 'diel_guilds.csv'), sep=',')
    plot_cre_dawn_dusk_peak_loc_bin_size_by_diel_guild(rootdir, all_peaks_df, feature, change_times_m, 'average', diel_guilds,
                                                       peak_feature='peak_loc', bin_size_min=30)

    # plot_cre_dawn_dusk_peak_loc(rootdir, all_peaks_df, feature, change_times_unit, name='average', peak_feature='peak_loc')



    # # get better look at the timing of peaks?
    # plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature, peak_feature='peak_amplitude')
    # plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature, peak_feature='peak')
    # plot_cre_dawn_dusk_peak_loc(rootdir, cres_peaks, feature, change_times_unit, name='average', peak_feature='peak_loc')
    # plot_cre_dawn_dusk_peak_loc(rootdir, cres_peaks_indiv, feature, change_times_unit, name='individual',
    #                             peak_feature='peak_loc')
    # plot_cre_dawn_dusk_stacked(rootdir, cres_peaks, feature, peak_feature='peak')
    #
    # # for plotting peaks of an individual species
