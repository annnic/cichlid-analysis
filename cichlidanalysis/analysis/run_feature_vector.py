import warnings
import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.io_feature_vector import load_feature_vectors, load_diel_pattern
from cichlidanalysis.io.io_ecological_measures import get_meta_paths
from cichlidanalysis.io.meta import extract_meta
from cichlidanalysis.utils.species_names import six_letter_sp_name
from cichlidanalysis.utils.species_metrics import tribe_cols
from cichlidanalysis.analysis.diel_pattern import daily_more_than_pattern_individ, daily_more_than_pattern_species, \
    day_night_ratio_individ, day_night_ratio_species
from cichlidanalysis.analysis.linear_regression import feature_correlations
from cichlidanalysis.analysis.bouts import names_bouts
from cichlidanalysis.analysis.ecological_als import diet_vs_size
from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg
from cichlidanalysis.plotting.figure_1 import cluster_dics
from cichlidanalysis.plotting.plot_total_rest import plot_total_rest_ordered, plot_total_rest_temporal, \
    plot_total_rest_diet, plot_total_rest_hist, plot_total_rest_vs_spd
from cichlidanalysis.plotting.plot_bouts import plot_bout_lens_rest_day_night, plot_dn_dif_rest_bouts, \
    plot_bout_lens_non_rest_day_night, plot_dn_dif_non_rest_bouts, rest_bouts_hists
from cichlidanalysis.plotting.plot_eco_traits import plot_total_rest_ecospace, plot_ecospace_vs_temporal_guilds, \
    plot_d15N_d13C_diet_guilds, plot_diet_guilds_hist, plot_total_rest_vs_diet_significance, \
    plot_ecospace_vs_temporal_guilds_density


# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def subset_feature_plt(rootdir, averages_i, features, plot_label, labelling):
    """

    :param averages_i:
    :param features:
    :param labelling:
    :return:
    """
    # fig = plt.figure(figsize=(5, 10))
    fig = sns.clustermap(averages_i.T.loc[:, features], col_cluster=False,
                         yticklabels=True)  # , cbar_kws=dict(use_gridspec=False,location="top")
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(rootdir, "subset_feature_plot_{}.png".format(plot_label)))

    plt.close()


def setup_feature_vector_data(rootdir):
    feature_v = load_feature_vectors(rootdir, "*als_fv2.csv")
    diel_patterns = load_diel_pattern(rootdir, suffix="*dp.csv")
    ronco_data_path, cichlid_meta_path = get_meta_paths()
    ronco_data = pd.read_csv(ronco_data_path)
    cichlid_meta = pd.read_csv(cichlid_meta_path)

    # add species_six
    feature_v['species_six'] = 'undefined'
    for id_n, id in enumerate(feature_v.fish_ID):
        sp = extract_meta(id)['species']
        feature_v.loc[id_n, 'species_six'] = six_letter_sp_name(sp)

    # # renaming misspelled species name
    feature_v = feature_v.replace('Aalcal', 'Altcal')
    # merge feature_v with ronco data and cichlid_meta
    feature_v = feature_v.merge(cichlid_meta, on="species_six")
    feature_v = feature_v.merge(diel_patterns.rename(columns={'species': 'six_letter_name_Ronco'}),
                                on="six_letter_name_Ronco")
    species = feature_v['six_letter_name_Ronco'].unique()

    # add column for cluster, hardcoded!!!!
    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    feature_v['cluster_pattern'] = 'placeholder'
    for key in dic_simple:
        # find the species which are in diel cluster group
        sp_diel_group = set(diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species'].to_list())
        feature_v.loc[feature_v.six_letter_name_Ronco.isin(sp_diel_group), 'cluster_pattern'] = key

    # make species average
    for species_n, species_name in enumerate(species):
        # get speeds for each individual for a given species
        sp_subset = feature_v[feature_v.six_letter_name_Ronco == species_name]

        # calculate ave and stdv
        average = sp_subset.mean(axis=0)
        average = average.append(pd.Series(sp_subset.cluster_pattern.unique()[0], index=['cluster_pattern']))
        average = average.rename(species_name)
        if species_n == 0:
            averages = average
        else:
            averages = pd.concat([averages, average], axis=1, join='inner')
        stdv = sp_subset.std(axis=0)

    return feature_v, averages, ronco_data, cichlid_meta, diel_patterns, species


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    rootdir = select_dir_path()

    feature_v, averages, ronco_data, cichlid_meta, diel_patterns, species = setup_feature_vector_data(rootdir)

    dic_complex, dic_simple, col_dic_simple, col_dic_complex, col_dic_simple, cluster_order = cluster_dics()
    tribe_col = tribe_cols()

    # histogram of total rest
    feature_v_mean = feature_v.groupby(['six_letter_name_Ronco', 'cluster_pattern']).mean()
    feature_v_mean = feature_v_mean.reset_index()

    feature_v_mean.loc[:, ['six_letter_name_Ronco', 'total_rest', 'peak_amplitude', 'peak', 'day_night_dif', 'cluster']
    ].to_csv(os.path.join(rootdir, "combined_cichlid_data_{}.csv".format(datetime.date.today())))
    print("Finished saving out species data")

    # combine behavioural data with the Ronco ecological data
    ave_rest = averages.loc[['total_rest', 'rest_mean_night', 'rest_mean_day', 'fish_length_mm', 'cluster_pattern'],
               :].transpose().reset_index().rename(
        columns={'index': 'sp'})
    ave_rest['night-day_dif_rest'] = ave_rest.rest_mean_night - ave_rest.rest_mean_day
    ave_rest['day-night_dif_rest'] = ave_rest.rest_mean_day - ave_rest.rest_mean_night
    sp_in_both = set(ave_rest.sp) & set(ronco_data.sp)
    missing_in_ronco = set(ave_rest.sp) - set(sp_in_both)
    feature_v_eco = pd.merge(ronco_data, ave_rest, how='left', on='sp')
    feature_v_eco = feature_v_eco.drop(feature_v_eco.loc[(pd.isnull(feature_v_eco.loc[:, 'total_rest']))].index).reset_index(drop=True)
    feature_v_eco = feature_v_eco.rename(columns={'sp': 'six_letter_name_Ronco'})
    feature_v_eco = pd.merge(feature_v_eco, cichlid_meta, how='left', on='six_letter_name_Ronco')

    # regression between features
    fv_eco_sp_ave = feature_v_eco.groupby(['six_letter_name_Ronco', 'cluster_pattern']).mean().reset_index('cluster_pattern')


    # ## heatmap of fv
    # averages_vals = averages.drop(averages[averages.isna().any(axis=1)].index)
    # rows = []
    # for i, element in enumerate(averages_vals.iloc[:, 0]):
    #     if isinstance(element, float):
    #         rows.append(i)
    # averages_vals = averages_vals.iloc[rows, :]
    # averages_norm = averages_vals.div(averages_vals.sum(axis=1), axis=0)

    # fig1, ax1 = plt.subplots()
    # fig1.set_figheight(6)
    # fig1.set_figwidth(12)
    # im_spd = ax1.imshow(averages_norm.T, aspect='auto', vmin=0, cmap='magma')
    # ax1.get_yaxis().set_ticks(np.arange(0, len(species)))
    # ax1.get_yaxis().set_ticklabels(averages_norm.columns, rotation=0)
    # ax1.get_xaxis().set_ticks(np.arange(0, averages_norm.shape[0]))

    # ax1.get_xaxis().set_ticklabels(averages_norm.index, rotation=90)
    # plt.title('Feature vector (normalised by feature)')
    # fig1.tight_layout(pad=3)

    # # clustered heatmap of  fv
    # fig = sns.clustermap(averages_norm, figsize=(20, 10), col_cluster=False, method='single', yticklabels=True)
    # plt.savefig(os.path.join(rootdir, "cluster_map_fv_{0}.png".format(datetime.date.today())))

    ### summary statistics ###
    # N per species histogram
    fig = plt.figure(figsize=(5, 5))
    ax = feature_v["six_letter_name_Ronco"].value_counts(sort=False).plot.hist()
    ax.set_xlabel("Individuals for a species")
    ax.set_xlim([0, 14])
    plt.savefig(os.path.join(rootdir, "Individuals_for_each_species.png"), dpi=1000)
    plt.close()

    # number of species
    # feature_v["six_letter_name_Ronco"].value_counts()

    ### plotting total rest plots ###
    plot_total_rest_ordered(rootdir, feature_v)
    plot_total_rest_temporal(rootdir, feature_v)
    plot_total_rest_diet(rootdir, feature_v)
    plot_total_rest_hist(rootdir, feature_v, feature_v_mean)
    plot_total_rest_vs_spd(rootdir, feature_v)

    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(data=feature_v, x='total_rest', y='spd_mean_night')
    plt.savefig(os.path.join(rootdir, "total_rest_vs_spd_mean_night.png"))
    plt.close()

    # fig = plt.figure(figsize=(5, 5))
    # sns.regplot(data=feature_v, x='total_rest', y='spd_max_mean')
    # plt.savefig(os.path.join(rootdir, "total_rest_vs_spd_max_mean.png"))
    # plt.close()

    plot_bout_lens_rest_day_night(rootdir, feature_v, diel_patterns)
    plot_dn_dif_rest_bouts(rootdir, feature_v, diel_patterns)
    plot_bout_lens_non_rest_day_night(rootdir, feature_v, diel_patterns)
    plot_dn_dif_non_rest_bouts(rootdir, feature_v, diel_patterns)

    feature_v_mean['rest_bout_mean_dn_dif'] = feature_v_mean['rest_bout_mean_day'] - feature_v_mean[
        'rest_bout_mean_night']
    feature_v_mean['nonrest_bout_mean_dn_dif'] = feature_v_mean['nonrest_bout_mean_day'] - feature_v_mean[
        'nonrest_bout_mean_night']
    rest_bouts_hists(rootdir, feature_v_mean)

    data_names, time_v2_m_names, spd_means, rest_means, move_means, rest_b_means, nonrest_b_means, move_b_means, nonmove_b_means = names_bouts()

    subset_feature_plt(rootdir, averages, spd_means, 'ave_spd_mmps', 'Average speed mm/s')
    subset_feature_plt(rootdir, averages, rest_means, 'ave_rest_ph', 'Average fraction rest per hour')
    subset_feature_plt(rootdir, averages, move_b_means, 'ave_move_bout_len', 'Average movement bout length')
    subset_feature_plt(rootdir, averages, nonmove_b_means, 'ave_non-move_bout_len', 'Average nonmovement bout length')
    subset_feature_plt(rootdir, averages, rest_b_means, 'ave_rest_bout_len',  'Average rest bout length')
    subset_feature_plt(rootdir, averages, nonrest_b_means, 'ave_non-rest_bout_len',  'Average nonrest bout length')

    ### Daily activity pattern ratio vs clusters ###
    diel_fish = daily_more_than_pattern_individ(feature_v, species, plot=False)
    diel_species = daily_more_than_pattern_species(averages, plot=False)
    day_night_ratio_fish = day_night_ratio_individ(feature_v)
    day_night_ratio_sp = day_night_ratio_species(averages)

    sns.clustermap(pd.concat([day_night_ratio_fish.ratio, diel_fish.diurnal * 1], axis=1), col_cluster=False,
                   yticklabels=day_night_ratio_fish.species, cmap='RdBu_r', vmin=0, vmax=2)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "clustermap_day_night_ratio_fish.png"))
    plt.close()

    sns.clustermap(pd.concat([day_night_ratio_sp, diel_species.diurnal * 1], axis=1), col_cluster=False,
                   cmap='RdBu_r',
                   vmin=0, vmax=2)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "clustermap_day_night_ratio_species.png"))
    plt.close()

    #### Correlations for average behaviour vs average species ecological measures
    x = feature_v_mean.total_rest
    y = feature_v_mean.fish_length_mm
    col_vector = cichlid_meta.set_index('six_letter_name_Ronco').loc[
        feature_v_mean.six_letter_name_Ronco.to_list(), 'diet'].reset_index().drop_duplicates().diet

    diet_vs_size(rootdir, col_vector, cichlid_meta, feature_v_mean)
    feature_correlations(rootdir, feature_v_mean, fv_eco_sp_ave)
    plot_total_rest_ecospace(rootdir, fv_eco_sp_ave, ronco_data)
    plot_ecospace_vs_temporal_guilds(rootdir, feature_v_eco, ronco_data, diel_patterns, dic_simple, col_dic_simple, fv_eco_sp_ave)
    # plot_ecospace_vs_temporal_guilds_density(rootdir, ronco_data, diel_patterns, dic_simple, col_dic_simple, fv_eco_sp_ave)
    plot_d15N_d13C_diet_guilds(rootdir, feature_v_eco, fv_eco_sp_ave, ronco_data)
    plot_diet_guilds_hist(rootdir, feature_v_eco, dic_simple, diel_patterns)
    plot_total_rest_vs_diet_significance(rootdir, feature_v_eco)

    # vertical position during the day vs total rest
    data1 = feature_v_mean.total_rest
    data2 = feature_v_mean.y_mean_day
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)

    # vertical position during the night vs total rest
    data1 = feature_v_mean.total_rest
    data2 = feature_v_mean.y_mean_night
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)

    # vp day vs night
    data1 = feature_v_mean.y_mean_day
    data2 = feature_v_mean.y_mean_night
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)

    # vertical position during the night vs total rest of diurnal clustered fish
    cluster_p = 'diurnal'
    data1 = feature_v_mean.loc[feature_v_mean.cluster_pattern == cluster_p, 'total_rest']
    data2 = feature_v_mean.loc[feature_v_mean.cluster_pattern == cluster_p, 'y_mean_day']
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq, cluster_p)
