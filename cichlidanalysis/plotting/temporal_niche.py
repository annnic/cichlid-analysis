import os

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec
import cmasher as cmr

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.analysis.processing import feature_daily
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.analysis.run_feature_vector import setup_feature_vector_data
from cichlidanalysis.analysis.run_binned_als import setup_run_binned


def plot_temporal_niche(rootdir, aves_ave_rest):
    ###### temporal niche by diet
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    _, _, _, cichlid_meta, diel_patterns, species = setup_feature_vector_data(rootdir)
    diets = ['Invertivore', 'Piscivore', 'Zooplanktivore', 'Algivore']
    my_palette = {'Invertivore': 'tomato', 'Piscivore': 'steelblue', 'Zooplanktivore': 'sandybrown',
                  'Algivore': 'mediumseagreen'}

    for diet_n, diet in enumerate(diets):
        select_sp = cichlid_meta.loc[cichlid_meta.diet == diet, 'six_letter_name_Ronco'].unique()
        select_data = aves_ave_rest.transpose().loc[list(set(aves_ave_rest.columns).intersection(select_sp))]
        select_data_inv = abs(select_data - 1)
        # rest_sum_norm = select_data.sum(axis=1)/select_data.shape[0]
        non_rest_max = select_data_inv.max(axis=0)
        non_rest_ave = select_data_inv.mean(axis=0)

        row_colors = pd.DataFrame(data=[my_palette[diet]] * len(select_data_inv.index.to_list()),
                                  index=select_data_inv.index.to_list())

        sns.clustermap(select_data_inv, col_cluster=False, cmap='cividis', vmin=0, vmax=1, method='ward',
                            figsize=(3.7, 3.7), row_colors=row_colors)
        ax = plt.gca()
        # ax.ax_row_dendrogram.set_visible(False)  # suppress row dendrogram
        plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_{}.png".format(diet)), dpi=350)
        plt.close()
        if diet_n == 0:
            all_non_rest_max = non_rest_max.to_frame(name=diet)
            all_non_rest_ave = non_rest_ave.to_frame(name=diet)

        else:
            all_non_rest_max = pd.concat([all_non_rest_max, non_rest_max.to_frame(name=diet)], axis=1)
            all_non_rest_ave = pd.concat([all_non_rest_ave, non_rest_ave.to_frame(name=diet)], axis=1)

        # rest_10 = select_data.quantile(0.1)
        # plt.figure(figsize=(4, 4))
        # # plt.plot(select_data.transpose().index, rest_min)
        # # plt.plot(select_data.transpose().index, rest_10)
        # plt.plot(select_data.transpose())
        # plt.ylim([0, 1])
        # plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_{}_test.png".format(diet)), dpi=350)
        # plt.close()

        # plt.figure(figsize=(4, 4))
        # for species in select_data_inv.index:
        #     plt.fill_between(select_data_inv.transpose().index, select_data_inv.loc[species], alpha=0.1, color=my_palette[diet])
        # plt.ylim([0, 1])
        # plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_{}_fill.png".format(diet)), dpi=350)
        # plt.close()

    row_colors_groups = pd.DataFrame.from_dict(my_palette, orient='index')
    sns.clustermap(all_non_rest_ave.transpose(), col_cluster=False, row_cluster=False, cmap='RdPu_r', vmin=0, vmax=1,
                   method='ward',
                   figsize=(3.7, 3.7), row_colors=row_colors_groups)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_mean.png"), dpi=350)
    plt.close()

    aves_ave_rest_t = aves_ave_rest.transpose()
    aves_ave_rest_t_ = aves_ave_rest_t.reset_index().rename(columns={'index': 'six_letter_name_Ronco'})
    aves_ave_rest_t_diet = pd.merge(aves_ave_rest_t_, cichlid_meta, how='left', on='six_letter_name_Ronco')

    aves_ave_rest_t_diet_no_nan = aves_ave_rest_t_diet.drop(
        aves_ave_rest_t_diet.loc[aves_ave_rest_t_diet['diet'].isnull()].index)
    aves_ave_rest_t_diet_no_nan = aves_ave_rest_t_diet_no_nan.set_index('six_letter_name_Ronco')
    aves_ave_rest_t_diet_no_nan = aves_ave_rest_t_diet_no_nan.sort_values('diet')

    # Prepare a vector of color mapped to the 'cyl' column
    # colors = ['mediumseagreen', 'sandybrown', 'tomato', 'steelblue']
    # diet_order = ['Algivore', 'Zooplanktivore', 'Invertivore', 'Piscivore']
    # my_palette = dict(zip(aves_ave_rest_t_diet_no_nan.diet.unique(), ['tomato', 'silver', 'steelblue', 'sandybrown', 'mediumseagreen']))
    row_colors = aves_ave_rest_t_diet_no_nan.diet.map(my_palette)

    sns.clustermap(aves_ave_rest_t_diet_no_nan.iloc[:, 0:48], row_cluster=False, col_cluster=False, row_colors=row_colors)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning.png"), dpi=350)
    return


def plot_temporal_niche_one(rootdir, aves_ave_rest, loadings):
    # plot one combined block with all diet types and the mean
    ###### temporal niche by diet
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    _, _, _, cichlid_meta, diel_patterns, species = setup_feature_vector_data(rootdir)
    diets = ['Invertivore', 'Piscivore', 'Zooplanktivore', 'Algivore']
    my_palette = {'Invertivore': 'tomato', 'Piscivore': 'steelblue', 'Zooplanktivore': 'sandybrown',
                  'Algivore': 'mediumseagreen'}

    for diet_n, diet in enumerate(diets):
        select_sp = cichlid_meta.loc[cichlid_meta.diet == diet, 'six_letter_name_Ronco'].unique()
        overlap = set(aves_ave_rest.columns.intersection(set(select_sp)))
        select_data = aves_ave_rest.transpose().loc[overlap]
        select_data_inv = abs(select_data - 1)
        non_rest_ave = select_data_inv.mean(axis=0)
        non_rest_90 = select_data_inv.quantile(q=0.9, axis=0)
        non_rest_std = select_data_inv.std(axis=0)

        row_colors = pd.DataFrame(data=[my_palette[diet]] * len(select_data_inv.index.to_list()),
                                  index=select_data_inv.index.to_list()).rename(columns={0: 'diet'})
        sns.clustermap(select_data_inv, col_cluster=False, cmap=cmr.neutral, vmin=0, vmax=1, method='ward',
                            figsize=(3.7, 3.7), row_colors=row_colors)
        plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_{}.pdf".format(diet)), dpi=350)
        plt.close()
        if diet_n == 0:
            all_non_rest_ave = non_rest_ave.to_frame(name=diet)
            all_non_rest_90 = non_rest_ave.to_frame(name=diet)
            all_non_rest_std = non_rest_std.to_frame(name=diet)

        else:
            all_non_rest_ave = pd.concat([all_non_rest_ave, non_rest_ave.to_frame(name=diet)], axis=1)
            all_non_rest_90 = pd.concat([all_non_rest_90, non_rest_90.to_frame(name=diet)], axis=1)
            all_non_rest_std = pd.concat([all_non_rest_std, non_rest_std.to_frame(name=diet)], axis=1)

    # group plot
    row_colors_groups = pd.DataFrame.from_dict(my_palette, orient='index').rename(columns={0: 'diet'})
    sns.clustermap(all_non_rest_ave.transpose(), col_cluster=False, row_cluster=False, cmap=cmr.neutral, vmin=0, vmax=1,
                   method='ward',
                   figsize=(3.7, 1), row_colors=row_colors_groups)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_mean.pdf"), dpi=350)
    plt.close()

    # species plot
    # need to order the dat first by 00:00, and then by diet group
    select_diet = cichlid_meta.loc[:, ['six_letter_name_Ronco', 'diet']].rename(columns={'six_letter_name_Ronco': 'species'}).drop_duplicates()
    # invert data
    aves_ave_rest_inv = abs(aves_ave_rest - 1)
    ordering_df = aves_ave_rest_inv.transpose().reset_index().rename(columns={'index': 'species'})

    ordering_df = ordering_df.merge(loadings.loc[:, ['species', 'pc1', 'pc2']], how='left', on='species')
    ordering_df = ordering_df.merge(select_diet, how='left', on='species')

    ordering_df = ordering_df.sort_values(by=['diet', '00:00']).set_index('species')
    row_colors_sp = ordering_df['diet'].map(my_palette)
    ordered_df = ordering_df.iloc[:, 0:48]

    sns.clustermap(ordered_df, col_cluster=False, row_cluster=False, cmap=cmr.neutral, vmin=0, vmax=1,
                   figsize=(3.7, 5), row_colors=row_colors_sp, yticklabels=1)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_one_big.pdf"), dpi=350)
    plt.close()

    # combined species and group plot
    species_and_groups = pd.concat([ordered_df, all_non_rest_ave.transpose()])
    row_colors_sp_and_groups = pd.concat([row_colors_sp, row_colors_groups])
    sns.clustermap(species_and_groups, col_cluster=False, row_cluster=False, cmap=cmr.neutral, vmin=0, vmax=1,
                   figsize=(2, 5.5), row_colors=row_colors_sp_and_groups, yticklabels=1)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_combined_mean.pdf"), dpi=350)
    plt.close()
    return


def plot_temporal_niche_one_five_blocks(rootdir, aves_ave_rest, loadings):
    ###### temporal niche by diet
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    _, _, _, cichlid_meta, diel_patterns, species = setup_feature_vector_data(rootdir)
    diets = ['Invertivore', 'Piscivore', 'Zooplanktivore', 'Algivore']
    my_palette = {'Invertivore': 'tomato', 'Piscivore': 'steelblue', 'Zooplanktivore': 'sandybrown',
                  'Algivore': 'mediumseagreen'}

    # collapse light states into four blocks: night1, dawn, day, dusk, night2
    aves_ave_rest_full = aves_ave_rest.reset_index()
    block_dic = {'night1': np.arange(0, 14), 'dawn': [14], 'day': np.arange(15, 37), 'dusk': [37],
                 'night2': np.arange(38, 48)}
    for epoch_n, epoch in enumerate(block_dic):
        ave_series = aves_ave_rest_full.iloc[block_dic[epoch], :].mean(axis=0).to_frame(name=epoch)
        if epoch_n == 0:
            aves_ave_rest_full_all = ave_series
        else:
            aves_ave_rest_full_all = pd.concat([aves_ave_rest_full_all, ave_series], axis=1)
    aves_ave_rest_full_all = aves_ave_rest_full_all.transpose()

    for diet_n, diet in enumerate(diets):
        select_sp = cichlid_meta.loc[cichlid_meta.diet == diet, 'six_letter_name_Ronco'].unique()
        overlap = set(aves_ave_rest_full_all.columns.intersection(set(select_sp)))
        select_data = aves_ave_rest_full_all.transpose().loc[overlap]
        select_data_inv = abs(select_data - 1)
        non_rest_ave = select_data_inv.mean(axis=0)
        non_rest_90 = select_data_inv.quantile(q=0.9, axis=0)

        row_colors = pd.DataFrame(data=[my_palette[diet]] * len(select_data_inv.index.to_list()),
                                  index=select_data_inv.index.to_list()).rename(columns={0: 'diet'})
        sns.clustermap(select_data_inv, col_cluster=False, cmap='cividis', vmin=0, vmax=1, method='ward',
                            figsize=(3.7, 3.7), row_colors=row_colors)
        plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_5blocks_{}.pdf".format(diet)), dpi=350)
        plt.close()
        if diet_n == 0:
            all_non_rest_ave = non_rest_ave.to_frame(name=diet)
            all_non_rest_90 = non_rest_ave.to_frame(name=diet)

        else:
            all_non_rest_ave = pd.concat([all_non_rest_ave, non_rest_ave.to_frame(name=diet)], axis=1)
            all_non_rest_90 = pd.concat([all_non_rest_90, non_rest_90.to_frame(name=diet)], axis=1)

    # group plot
    row_colors_groups = pd.DataFrame.from_dict(my_palette, orient='index').rename(columns={0: 'diet'})
    sns.clustermap(all_non_rest_ave.transpose(), col_cluster=False, row_cluster=False, cmap='cividis', vmin=0, vmax=1,
                   method='ward',
                   figsize=(3.7, 1), row_colors=row_colors_groups)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_5blocks_mean.pdf"), dpi=350)
    plt.close()

    # species plot
    # need to order the dat first by 00:00, and then by diet group
    select_diet = cichlid_meta.loc[:, ['six_letter_name_Ronco', 'diet']].rename(columns={'six_letter_name_Ronco': 'species'}).drop_duplicates()
    # invert data
    aves_ave_rest_inv = abs(aves_ave_rest_full_all - 1)
    ordering_df = aves_ave_rest_inv.transpose().reset_index().rename(columns={'index': 'species'})

    ordering_df = ordering_df.merge(loadings.loc[:, ['species', 'pc1', 'pc2']], how='left', on='species')
    ordering_df = ordering_df.merge(select_diet, how='left', on='species')

    ordering_df = ordering_df.sort_values(by=['diet', 'day']).set_index('species')
    row_colors_sp = ordering_df['diet'].map(my_palette)
    ordered_df = ordering_df.iloc[:, 0:5]

    sns.clustermap(ordered_df, col_cluster=False, row_cluster=False, cmap='cividis', vmin=0, vmax=1,
                   figsize=(3.7, 5), row_colors=row_colors_sp, yticklabels=1)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_one_big_5blocks_day_sorted.pdf"), dpi=350)
    plt.close()

    # combined species and group plot
    species_and_groups = pd.concat([ordered_df, all_non_rest_90.transpose()])
    row_colors_sp_and_groups = pd.concat([row_colors_sp, row_colors_groups])
    sns.clustermap(species_and_groups, col_cluster=False, row_cluster=False, cmap='cividis', vmin=0, vmax=1,
                   figsize=(3.7, 5), row_colors=row_colors_sp_and_groups, yticklabels=1)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_combined_90pc_5blocks.pdf"), dpi=350)
    plt.close()
    return


if __name__ == '__main__':
    rootdir = select_dir_path()

    loadings = pd.read_csv(os.path.join(rootdir, 'pca_loadings.csv'))

    fish_tracks_bin, sp_metrics, tribe_col, species_full, fish_IDs, species_sixes = setup_run_binned(rootdir)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, \
    day_ns, day_s, change_times_d, change_times_m, change_times_datetime, change_times_unit \
        = load_timings(fish_tracks_bin[fish_tracks_bin.FishID == fish_IDs[0]].shape[0])
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    # ## ridge plots and averages for each feature ###
    averages_vp, date_time_obj_vp, sp_vp_combined, averages_spd, sp_spd_combined, averages_rest, sp_rest_combined, \
    averages_move, sp_move_combined = plot_ridge_plots(fish_tracks_bin, change_times_datetime,
                                                       rootdir, sp_metrics, tribe_col)

    aves_ave_rest = feature_daily(averages_rest)

    # plot_temporal_niche(rootdir, aves_ave_rest)
    plot_temporal_niche_one(rootdir, aves_ave_rest, loadings)

    # sp_to_tribes = sp_metrics.loc[:, ['tribe', 'six_letter_name_Ronco']].rename(columns={"six_letter_name_Ronco": "species"})
    # plot_temporal_niche_one(rootdir, aves_ave_rest, loadings, tribe_col, sp_to_tribes)
