import warnings
import os

import datetime as dt

import pandas as pd

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.als_files import load_bin_als_files
from cichlidanalysis.io.io_ecological_measures import get_meta_paths
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_metrics import tribe_cols
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import diel_pattern_stats_individ_bin, diel_pattern_stats_species_bin
from cichlidanalysis.analysis.self_correlations import species_daily_corr, fish_daily_corr, fish_weekly_corr, \
    plot_corr_coefs, get_corr_coefs_daily, intra_individ_corr
from cichlidanalysis.analysis.crepuscular_pattern import crepuscular_peaks, crespuscular_weekly_fish, \
    crespuscular_daily_ave_fish
from cichlidanalysis.analysis.clustering_patterns import run_species_pattern_cluster_daily, \
    run_species_pattern_cluster_weekly
from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg
from cichlidanalysis.plotting.cluster_plots import cluster_all_fish, cluster_species_daily
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.plotting.figure_1 import cluster_daily_ave, clustered_spd_map, cluster_dics, plot_all_spd_subplots
from cichlidanalysis.plotting.position_plots import plot_combined_v_position
from cichlidanalysis.plotting.plot_diel_patterns import plot_day_night_species, plot_cre_dawn_dusk_strip_box, \
    plot_day_night_species_ave, plot_cre_dawn_dusk_stacked, plot_cre_dawn_dusk_peak_loc

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_run_binned(rootdir, als_type='*als_30m.csv'):
    fish_tracks_bin_i = load_bin_als_files(rootdir, als_type)
    fish_tracks_bin_i = fish_tracks_bin_i.reset_index(drop=True)
    fish_tracks_bin_i['time_of_day_dt'] = fish_tracks_bin_i.ts.apply(
        lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    _, cichlid_meta_path = get_meta_paths()
    sp_metrics = pd.read_csv(cichlid_meta_path)

    # getting extra data (colours for plotting, species metrics)
    tribe_col = tribe_cols()

    # add species six names, tribe and other meta data
    fish_tracks_bin_i = fish_tracks_bin_i.rename(columns={"species": "species_our_names"})
    fish_tracks_bin_i = fish_tracks_bin_i.merge(sp_metrics, on='species_our_names')
    fish_tracks_bin_i = add_day_number_fish_tracks(fish_tracks_bin_i)
    fish_tracks_bin_i = fish_tracks_bin_i.rename(columns={"six_letter_name_Ronco": "species"})

    # get each fish ID and all species
    fish_IDs = fish_tracks_bin_i['FishID'].unique()
    species_true = fish_tracks_bin_i['species_our_names'].unique()
    species_sixes = fish_tracks_bin_i['species'].unique()

    return fish_tracks_bin_i, sp_metrics, tribe_col, species_true, fish_IDs, species_sixes


if __name__ == '__main__':
    rootdir = select_dir_path()

    fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes = setup_run_binned(rootdir)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit \
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # ###########################
    # ## ridge plots and averages for each feature ###
    averages_vp, date_time_obj_vp, sp_vp_combined, averages_spd, sp_spd_combined, averages_rest, sp_rest_combined, \
    averages_move, sp_move_combined = plot_ridge_plots(fish_tracks_bin, change_times_datetime,
                                                       rootdir, sp_metrics, tribe_col)

    # ### generate averages of the averages ###
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_vp = feature_daily(averages_vp)
    aves_ave_rest = feature_daily(averages_rest)
    aves_ave_move = feature_daily(averages_move)

    aves_ave_spd.columns = species_sixes
    aves_ave_vp.columns = species_sixes
    aves_ave_rest.columns = species_sixes
    aves_ave_move.columns = species_sixes


    # ###########################
    ## correlations ##
    # correlations for days across week for an individual - intra-individual variability
    mean_corr_per_fish = intra_individ_corr(rootdir, fish_tracks_bin, 'speed_mm')

    features = ['speed_mm', 'rest']
    for feature in features:
        # correlations for individuals of species across daily average of feature - daily inter-individual variability
        corr_vals_long = get_corr_coefs_daily(rootdir, fish_tracks_bin, feature, species_sixes)
        plot_corr_coefs(rootdir, corr_vals_long, feature, 'daily')

        # correlations for individuals across week - weekly inter-individual variability
        corr_vals_long_weekly = fish_weekly_corr(rootdir, fish_tracks_bin, feature, 'single', False)
        plot_corr_coefs(rootdir, corr_vals_long_weekly, feature, 'weekly')

    # save out corr_vals
    mean_corr_per_fish.to_csv(os.path.join(rootdir, 'mean_corr_per_fish.csv'), sep=',', index=False, encoding='utf-8')
    corr_vals_long.to_csv(os.path.join(rootdir, 'corr_vals_long_daily.csv'), sep=',', index=False, encoding='utf-8')
    corr_vals_long_weekly.to_csv(os.path.join(rootdir, 'corr_vals_long_weekly.csv'), sep=',', index=False,
                                 encoding='utf-8')

    # ### correlations for species and clusters ####
    run = False
    if run:
        species_daily_corr(rootdir, aves_ave_spd, 'ave', 'speed_mm', 'single')
        species_daily_corr(rootdir, aves_ave_rest, 'ave', 'rest', 'single')
        species_daily_corr(rootdir, aves_ave_move, 'ave', 'movement', 'single')
        species_daily_corr(rootdir, aves_ave_vp, 'ave', 'vertical', 'single')

    # clustered patterns of daily activity
    species_cluster_spd, species_cluster_move, species_cluster_rest = run_species_pattern_cluster_daily(aves_ave_spd,
                                                                                                        aves_ave_move,
                                                                                                        aves_ave_rest,
                                                                                                        aves_ave_vp,
                                                                                                        rootdir)
    # species_cluster_spd_wk, species_cluster_move_wk, species_cluster_rest_wk = run_species_pattern_cluster_weekly(
    #     averages_spd, averages_move, averages_rest, rootdir)

    # Figures of clustered corr matrix and cluster average speed: figure 1
    clustered_spd_map(rootdir, aves_ave_spd, link_method='single', max_d=1.35)
    cluster_daily_ave(rootdir, aves_ave_spd, 'speed', link_method='single', max_d=1.35)
    cluster_daily_ave(rootdir, aves_ave_vp, 'vertical', link_method='single', max_d=2)

    #### heatmap
    # cmap = plt.get_cmap('bwr')
    # min_val = fish_diel_patterns_sp.day_night_dif.min()
    # max_val = fish_diel_patterns_sp.day_night_dif.max()
    # min_val = -max_val
    # scaled_data = (fish_diel_patterns_sp.day_night_dif - min_val) / (max_val - min_val)
    # row_colors = cmap(scaled_data)
    # g= sns.clustermap(aves_ave_spd.transpose(), row_cluster=True, col_cluster=False, cmap='Blues',
    #                row_colors=row_colors, yticklabels=True, z_score=1)
    # plt.savefig(os.path.join(rootdir, "figure_panel_1_testing.png"))


    # ###########################
    # ### Define and plot diel pattern for each type ###
    fish_tracks_bin = fish_tracks_add_day_twilight_night(fish_tracks_bin)
    fish_diel_patterns = diel_pattern_stats_individ_bin(fish_tracks_bin, feature='rest')
    fish_diel_patterns_sp = diel_pattern_stats_species_bin(fish_tracks_bin, feature='rest')
    plot_day_night_species_ave(rootdir, fish_diel_patterns, fish_diel_patterns_sp, feature='rest')

    fish_diel_patterns_move = diel_pattern_stats_individ_bin(fish_tracks_bin, feature='movement')
    fish_diel_patterns_move_sp = diel_pattern_stats_species_bin(fish_tracks_bin, feature='movement')
    plot_day_night_species_ave(rootdir, fish_diel_patterns_move, fish_diel_patterns_move_sp, feature="movement")

    fish_diel_patterns_spd = diel_pattern_stats_individ_bin(fish_tracks_bin, feature='speed_mm')
    fish_diel_patterns_spd_sp = diel_pattern_stats_species_bin(fish_tracks_bin, feature='speed_mm')
    plot_day_night_species_ave(rootdir, fish_diel_patterns_spd, fish_diel_patterns_spd_sp, feature='speed_mm')

    # finding the crepuscular features
    # feature = 'rest'
    # crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_bin, species)  # for plotting daily average for each species
    # crespuscular_weekly_fish(rootdir, feature, fish_tracks_bin, species)     # for plotting weekly data for each species

    feature = 'speed_mm'
    # get better look at the timing of peaks?
    cres_peaks, cres_peaks_indiv = crepuscular_peaks(rootdir, feature, fish_tracks_bin, fish_diel_patterns_sp)
    plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature, peak_feature='peak_amplitude')
    plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks, feature, peak_feature='peak')
    plot_cre_dawn_dusk_peak_loc(rootdir, cres_peaks, feature, change_times_unit, name='average', peak_feature='peak_loc')
    plot_cre_dawn_dusk_peak_loc(rootdir, cres_peaks_indiv, feature, change_times_unit, name='individual',
                                peak_feature='peak_loc')
    plot_cre_dawn_dusk_stacked(rootdir, cres_peaks, feature, peak_feature='peak')

    # for plotting peaks of an individual species
    crespuscular_daily_ave_fish(rootdir, feature, fish_tracks_bin, ['Astbur'])
    crespuscular_weekly_fish(rootdir, feature, fish_tracks_bin, ['Astbur'])

    # include = ['Neosav', 'Neooli', 'Neopul', 'Neohel', 'Neobri', 'Neocra', 'Neomar', 'NeofaM', "Neogra", 'Neocyg',
    #            'Neowal', 'Neofal']
    # cres_peaks_princess = cres_peaks[cres_peaks['species'].isin(include)]
    # plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks_princess, feature, peak_feature='peak')
    # plot_cre_dawn_dusk_strip_box(rootdir, cres_peaks_princess, feature, peak_feature='peak_amplitude')
    # plot_cre_dawn_dusk_stacked(rootdir, cres_peaks_princess, feature, peak_feature='peak')


    # make and save diel patterns csv
    cresp_sp = cres_peaks.groupby(['species']).mean()
    diel_sp = fish_diel_patterns.groupby('species').mean()
    diel_patterns_df = pd.concat([cresp_sp, diel_sp.day_night_dif], axis=1).reset_index()
    diel_patterns_df = diel_patterns_df.merge(species_cluster_spd, on="species")
    # add total rest

    diel_patterns_df.to_csv(os.path.join(rootdir, "combined_diel_patterns_{}_dp.csv".format(dt.date.today())))
    print("Finished saving out diel pattern data")

    # add column for cluster, hardcoded!!!!
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    fish_tracks_bin['cluster_pattern'] = 'placeholder'
    for key in dic_simple:
        # find the species which are in diel cluster group
        sp_diel_group = set(diel_patterns_df.loc[diel_patterns_df.cluster.isin(dic_simple[key]), 'species'].to_list())
        fish_tracks_bin.loc[fish_tracks_bin.species.isin(sp_diel_group), 'cluster_pattern'] = key


    # Correlations
    # peak fraction
    cres_peaks_ave = cres_peaks.groupby(by=['species', 'twilight']).mean().reset_index()
    data1 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dawn', ['peak']].reset_index(drop=True)
    data2 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dusk', ['peak']].reset_index(drop=True)
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1.peak, data2.peak, model, r_sq)

    # data1 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dawn', ['peak']].reset_index(drop=True)
    # data2 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dusk', ['peak']].reset_index(drop=True)
    # model, r_sq = run_linear_reg(data1.peak, diel_sp.day_night_dif)
    # plt_lin_reg(rootdir, data1.peak, diel_sp.loc[:, 'day_night_dif'].reset_index(drop=True), model, r_sq)
    #
    # model, r_sq = run_linear_reg(data1.peak, diel_sp.day_night_dif)
    # plt_lin_reg(rootdir, data1.peak, diel_sp.loc[:, 'day_night_dif'].reset_index(drop=True), model, r_sq)

    # cres_peaks_ave = cres_peaks.groupby(by=['species']).mean().reset_index()
    # data1 = cres_peaks_ave.peak
    # data2 = feature_v_mean.total_rest# feature_v_mean.day_night_dif
    # model, r_sq = run_linear_reg(data1, data2)
    # plt_lin_reg(rootdir, data1.peak, data2.peak, model, r_sq)

    # peak location timing
    data1 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dawn', ['peak_loc']].reset_index(drop=True)
    data2 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dusk', ['peak_loc']].reset_index(drop=True)
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1.peak_loc, data2.peak_loc, model, r_sq)

    # peak location timing dawn vs d/n pref
    data1 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dawn', ['peak_loc']].reset_index(drop=True)
    model, r_sq = run_linear_reg(data1, diel_sp.day_night_dif)
    plt_lin_reg(rootdir, data1.peak_loc, diel_sp.loc[:, 'day_night_dif'].reset_index(drop=True), model, r_sq, 'dawn')

    # peak location timing dusk vs d/n pref
    data1 = cres_peaks_ave.loc[cres_peaks_ave.twilight == 'dusk', ['peak_loc']].reset_index(drop=True)
    model, r_sq = run_linear_reg(data1, diel_sp.day_night_dif)
    plt_lin_reg(rootdir, data1.peak_loc, diel_sp.loc[:, 'day_night_dif'].reset_index(drop=True), model, r_sq, 'dusk')


    # ## feature vs time of day density plot
    # ax = sns.displot(pd.melt(aves_ave_move.reset_index(), id_vars='time_of_day'), x='time_of_day', y='value')
    # for axes in ax.axes.flat:
    #     _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    plot_combined_v_position(rootdir, fish_tracks_bin, fish_diel_patterns)

    # save out table S1
    fish_tracks_bin_table = fish_tracks_bin.loc[:, ['FishID', 'sex', 'species_true', 'species', 'tribe', 'diet']].drop_duplicates()

    sexes = fish_tracks_bin_table.groupby("species")['sex'].value_counts().unstack(fill_value=0).reset_index()
    sexes['fish_N'] = sexes.sum(axis=1)
    fish_tracks_bin_table_sp = fish_tracks_bin.loc[:, ['species_true', 'species', 'tribe', 'diet']].drop_duplicates().reset_index(drop=True)
    table1 = pd.merge(fish_tracks_bin_table_sp, sexes, on='species')
    table1.to_csv(os.path.join(rootdir, 'table_1.csv'), sep=',', index=False, encoding='utf-8')

