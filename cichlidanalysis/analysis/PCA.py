import copy
import os
from tkinter import *
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg
from cichlidanalysis.analysis.processing import feature_daily
from cichlidanalysis.analysis.run_binned_als import setup_run_binned
from cichlidanalysis.analysis.run_feature_vector import setup_feature_vector_data
from cichlidanalysis.io.io_feature_vector import load_diel_pattern
from cichlidanalysis.plotting.plot_pca import plot_loadings, plot_2D_pc_space, plot_variance_explained, \
    plot_factor_loading_matrix, pc_loadings_on_2D, plot_reconstruct_pc, plot_3D_pc_space, \
    plot_2D_pc_space_label_species, plot_2D_pc_space_colour, plot_pc, plot_2D_pc_space_orig, plot_norm_traces, \
    plot_temporal_pcs
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.plotting.figure_1 import plot_all_spd_subplots

# inspired by https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# and https://builtin.com/machine-learning/pca-in-python


def reorganise_behav(fish_tracks_bin, feature, feature_id, row_id='FishID', col_id='time_of_day_dt'):
    """ This takes in fish_tracks_bin data and reorganises so that each row is a fish, each column is a time bin and the
     values are the daily average of a feature. This is used for PCA analysis

    :param fish_tracks_bin:
    :param feature:
    :param feature_id:
    :param row_id:
    :param col_id:
    :return:
    """
    subset = fish_tracks_bin.loc[:, [row_id, col_id, feature, 'day_n']]
    subset_2 = subset.groupby([row_id, col_id]).mean()
    subset_3 = subset_2.reset_index().drop(columns='day_n')
    feature_reorg = subset_3.pivot(index=row_id, columns=col_id).add_prefix(feature_id).droplevel(0, axis=1)
    return feature_reorg


def fish_bin_pca_df(fish_tracks_bin, ronco_data):
    # reshape fish_tracks_bin by making FishID the rows
    org_spd = reorganise_behav(fish_tracks_bin, feature='speed_mm', feature_id='spd_', row_id='FishID',
                               col_id='time_of_day_dt')
    org_rest = reorganise_behav(fish_tracks_bin, feature='rest', feature_id='rest_', row_id='FishID',
                                col_id='time_of_day_dt')
    org_vp = reorganise_behav(fish_tracks_bin, feature='vertical_pos', feature_id='vp_', row_id='FishID',
                              col_id='time_of_day_dt')

    pca_df = pd.concat([org_spd, org_rest, org_vp], axis=1)  # .reset_index()
    to_add = fish_tracks_bin.loc[:, ['FishID', 'sex', 'size_male', 'size_female', 'habitat', 'diet', 'species']
             ].drop_duplicates().reset_index(drop=True)
    combined = to_add.merge(ronco_data.rename(columns={"sp": "species"}), 'left', on='species')
    to_add_ronco = combined.groupby(['FishID'], as_index=False).agg(
        {'species': 'first', 'sex': 'first', 'size_male': 'mean'
            , 'size_female': 'mean', 'habitat': 'first',
         'diet': 'first', 'body_PC1': 'mean', 'body_PC2': 'mean', 'LPJ_PC1': 'mean', 'LPJ_PC2': 'mean',
         'oral_PC1': 'mean', 'oral_PC2': 'mean', 'd15N': 'mean', 'd13C': 'mean'})
    targets = to_add_ronco.species
    to_add_ronco = to_add_ronco.drop(columns=['species', 'size_male', 'size_female'])
    pca_df = pca_df.reset_index().merge(to_add_ronco, how='left', on='FishID')
    pca_df['species'] = targets

    pca_df, df_key = replace_cat_with_nums(pca_df, col_names=['sex', 'habitat', 'diet'])
    pca_df = pca_df.dropna()
    targets = pca_df.species
    pca_df = pca_df.drop(columns=['species']).set_index('FishID')
    return pca_df, targets


def fish_fv_pca_df(feature_v, ronco_data):
    fv = feature_v.drop(columns=['parental_care', 'sociality', 'distribution', 'comments/links', 'size_male',
                                        'size_female', 'diet_contents', 'breeding', 'mouth_brooder', 'monogomous',
                                        'habitat_details', 'genome', 'tribe', 'notes', 'species_six', 'fish_n',
                                        'species_true', 'species_our_names', 'cluster'])

    combined = fv.merge(ronco_data.rename(columns={"sp": "six_letter_name_Ronco"}), 'left', on='six_letter_name_Ronco')
    ronco_to_fish = combined.groupby(['fish_ID'], as_index=False).agg(
        {'six_letter_name_Ronco': 'first', 'habitat': 'first',
         'diet': 'first', 'body_PC1': 'mean', 'body_PC2': 'mean', 'LPJ_PC1': 'mean', 'LPJ_PC2': 'mean',
         'oral_PC1': 'mean', 'oral_PC2': 'mean', 'd15N': 'mean', 'd13C': 'mean'})
    ronco_to_fish = ronco_to_fish.drop(columns=['six_letter_name_Ronco', 'habitat', 'diet'])
    pca_df_fv = fv.merge(ronco_to_fish, 'left', on='fish_ID')

    pca_df_fv, df_key = replace_cat_with_nums(pca_df_fv, col_names=['habitat', 'diet', 'cluster_pattern'])
    pca_df_fv = pca_df_fv.dropna()
    # drop undesired cols pf move and bout
    pca_df_fv = pca_df_fv.drop(columns=pca_df_fv.columns[pca_df_fv.columns.str.contains('move')])
    pca_df_fv = pca_df_fv.drop(columns=pca_df_fv.columns[pca_df_fv.columns.str.contains('bout')])

    # for species
    pca_df_fv_sp = pca_df_fv.groupby(by='six_letter_name_Ronco').mean()

    targets = pca_df_fv.six_letter_name_Ronco
    all_targets = pca_df_fv.loc[:, ['six_letter_name_Ronco', 'habitat', 'diet', 'cluster_pattern']]
    pca_df_fv = pca_df_fv.drop(columns=['six_letter_name_Ronco']).set_index('fish_ID')
    return pca_df_fv, targets, all_targets, df_key, pca_df_fv_sp


def run_pca(rootdir, data_input, norm='zscore', n_com=10):
    """ Takes a 2D pandas df, z-scores to standardise the data, runs PCA with n_com data

    :param rootdir: dir, here to save
    :param data_input: 2D matrix,
    :param norm: minmax or zscore
    :param n_com:
    :return:
    """

    # check that there's no nans
    if np.max(np.max(data_input.isnull())):
        print('Some nulls in the data, cannot run')
        return

    if norm == 'minmax':
        # scale data between 0 and 1
        min_max_scaler = MinMaxScaler()
        x = min_max_scaler.fit_transform(data_input.values)
        data_input_norm = pd.DataFrame(x, columns=data_input.columns, index=data_input.index)
    elif norm == 'zscore':
        # z = (x - u) / s
        # Standardizing the features -> is therefore covariance (if not scaled would be correlation)
        x = StandardScaler().fit_transform(data_input.values)
        data_input_norm = pd.DataFrame(x, columns=data_input.columns, index=data_input.index)

    mu = np.mean(data_input, axis=0)

    # run PCA
    pca = PCA(n_components=n_com)
    principalComponents = pca.fit_transform(x)
    labels = []
    for i in range(n_com):
        labels.append('pc{}'.format(i + 1))

    principalDf = pd.DataFrame(data=principalComponents, columns=labels)
    finalDf = pd.concat([principalDf, data_input.index.to_series().reset_index(drop=True)], axis=1)

    if norm == 'zscore':
        # reconstructing the fish series
        # https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
        Xhat = np.dot(pca.transform(data_input)[:, :n_com], pca.components_[:n_com, :])
        Xhat += mu
        reconstructed = pd.DataFrame(data=Xhat, columns=data_input.columns)
        f, ax = plt.subplots(figsize=(10, 5))
        plt.plot(reconstructed)
        plt.savefig(os.path.join(rootdir, "reconstructed.png"), dpi=1000)
        plt.close()

    # plot reconstruction of pc 'n'
    plot_reconstruct_pc(rootdir, data_input, pca, mu, 1)

    # plot loadings of each pc
    loadings = plot_loadings(rootdir, pca, labels, data_input)

    # f, ax = plt.subplots(figsize=(10, 5))
    # plt.plot(reconstructed.loc[:, 'Astbur'])
    # plt.savefig(os.path.join(rootdir, "reconstructed_Astbur.png"), dpi=1000)
    # plt.close()

    return pca, labels, loadings, finalDf, principalComponents, data_input_norm


def replace_cat_with_nums(df, col_names):
    """ As categorical values can't be used for PCA, they are converted to numbers, also saves out the key
    (could improve the format but need to see how it's used)"""
    # first test if every column is there:
    for col in col_names:
        if not col in df.columns:
            print('missing column {}, take out?'.format(col))
            return

    df_i = copy.copy(df)
    col_vals_all = []
    col_nums_all = []
    for col in col_names:
        col_vals = df_i.loc[:, col].unique().tolist()
        col_nums = np.arange(len(df_i.loc[:, col].unique())).tolist()
        df_i[col].replace(col_vals, col_nums, inplace=True)
        col_vals_all.extend(col_vals)
        col_nums_all.extend(col_nums)
    d = {'col_vals': col_vals_all, 'col_nums': col_nums_all}
    df_key = pd.DataFrame(data=d)
    return df_i, df_key


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes = setup_run_binned(rootdir)
    feature_v, averages, ronco_data, cichlid_meta, diel_patterns, species = setup_feature_vector_data(rootdir)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit \
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])

    averages_vp, date_time_obj_vp, sp_vp_combined, averages_spd, sp_spd_combined, averages_rest, sp_rest_combined, \
    averages_move, sp_move_combined = plot_ridge_plots(fish_tracks_bin, change_times_datetime,
                                                       rootdir, sp_metrics, tribe_col)
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_rest = feature_daily(averages_rest)

    for species_n, species_name in enumerate(averages_spd.drop(['time_of_day'], axis=1).columns):
        # get speeds for each individual for a given species
        feature = 'speed_mm'
        feature_i = fish_tracks_bin[fish_tracks_bin.species == species_name][[feature, 'FishID', 'ts']]
        sp_feature = feature_i.pivot(columns='FishID', values=feature, index='ts')

        sp_feature_daily = feature_daily(sp_feature)
        if species_n == 0:
            sp_feature_combined_weekly = sp_feature
            sp_feature_combined_daily = sp_feature_daily
        else:
            frames = [sp_feature_combined_weekly, sp_feature]
            sp_feature_combined = pd.concat(frames, axis=1)

            frames = [sp_feature_combined_daily, sp_feature_daily]
            sp_feature_combined_daily = pd.concat(frames, axis=1)

    diel_patterns = load_diel_pattern(rootdir, suffix="*dp.csv")

    ########### PCA matrix setup
    # pca_df, targets = fish_bin_pca_df(fish_tracks_bin, ronco_data)
    # pca_df_fv, targets, all_targets, df_key, pca_df_fv_sp = fish_fv_pca_df(feature_v, ronco_data)

    ########### other preprocessed/setups
    # # log transform
    # aves_ave_spd_log = np.log(aves_ave_spd)
    # aves_ave_spd.transpose() = ts as feature
    # aves_ave_spd = species as features
    # sp_feature_combined_daily = individual fish as features

    # aves_ave_spd with zscore is the input used for the paper
    run_pca_df = aves_ave_spd
    norm_method = 'zscore'
    pca, labels, loadings, finalDf, principalComponents, data_input_norm = run_pca(rootdir, run_pca_df, norm=norm_method)

    # plot normalised data input
    plot_norm_traces(rootdir, data_input_norm, norm_method)

    # independent plots
    plot_3D_pc_space(rootdir, run_pca_df, finalDf, pca)
    plot_factor_loading_matrix(rootdir, loadings, top_pc=2)
    pc_loadings_on_2D(rootdir, principalComponents[:, 0:2], np.transpose(pca.components_[0:2, :]), loadings, top_n=3)
    plot_pc(rootdir, finalDf, list_pcs=['pc1', 'pc2'])

    # figure 1
    plot_variance_explained(rootdir, pca)
    plot_2D_pc_space_orig(rootdir, run_pca_df, finalDf)
    plot_temporal_pcs(rootdir, finalDf, change_times_datetime)

    # just species.t aves_ave_spd_means, where species are the features
    averages_t = averages.transpose()
    averages_t = averages_t.reset_index().rename(columns={'index': 'species'})
    loadings_sp = loadings.reset_index().rename(columns={'index': 'species'})
    loadings_sp = loadings_sp.merge(averages_t[['species', 'day_night_dif', 'peak']], on='species', how='left')

    # save data:
    loadings_sp.to_csv(os.path.join(rootdir, 'pca_loadings.csv'), sep=',', index=False, encoding='utf-8')

    # pc1 vs day_night_dif
    model, r_sq = run_linear_reg(loadings_sp.day_night_dif.astype(float), loadings_sp.pc1)
    plt_lin_reg(rootdir, loadings_sp.day_night_dif.astype(float), loadings_sp.pc1, model, r_sq)

    # pc2 vs peak
    model, r_sq = run_linear_reg(loadings_sp.peak.astype(float), loadings_sp.pc2)
    plt_lin_reg(rootdir, loadings_sp.peak.astype(float), loadings_sp.pc2, model, r_sq)

    plot_all_spd_subplots(rootdir, fish_tracks_bin, change_times_datetime, loadings_sp)

