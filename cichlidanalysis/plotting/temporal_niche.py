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


def plot_temporal_niche_one(rootdir, aves_ave_rest, loadings, diel_guilds):
    # plot one combined block with all diet types and the mean
    ###### temporal niche by diet
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    _, _, _, cichlid_meta, diel_patterns, species = setup_feature_vector_data(rootdir)
    diets = ['Invertivore', 'Piscivore', 'Zooplanktivore', 'Algivore']
    my_palette = {'Invertivore': 'tomato', 'Piscivore': 'steelblue', 'Zooplanktivore': 'sandybrown',
                  'Algivore': 'mediumseagreen'}

    my_palette_diel = {'Diurnal': '#CED926', 'Nocturnal': '#40A9BF', 'Crepuscular': '#26D97A', 'Cathemeral': '#737F8C'}

    for diet_n, diet in enumerate(diets):
        select_sp = cichlid_meta.loc[cichlid_meta.diet == diet, 'six_letter_name_Ronco'].unique()
        overlap = set(aves_ave_rest.columns.intersection(set(select_sp)))
        select_data = aves_ave_rest.transpose().loc[overlap]
        # select_data_inv = abs( - 1)
        rest_ave = select_data.mean(axis=0)
        rest_90 = select_data.quantile(q=0.9, axis=0)
        rest_std = select_data.std(axis=0)

        row_colors = pd.DataFrame(data=[my_palette[diet]] * len(select_data.index.to_list()),
                                  index=select_data.index.to_list()).rename(columns={0: 'diet'})
        sns.clustermap(select_data, col_cluster=False, cmap=cmr.neutral, vmin=0, vmax=1, method='ward',
                            figsize=(3.7, 3.7), row_colors=row_colors)
        plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_{}.pdf".format(diet)), dpi=350)
        plt.close()
        if diet_n == 0:
            all_rest_ave = rest_ave.to_frame(name=diet)
            all_rest_90 = rest_ave.to_frame(name=diet)
            all_rest_std = rest_std.to_frame(name=diet)

        else:
            all_rest_ave = pd.concat([all_rest_ave, rest_ave.to_frame(name=diet)], axis=1)
            all_rest_90 = pd.concat([all_rest_90, rest_90.to_frame(name=diet)], axis=1)
            all_rest_std = pd.concat([all_rest_std, rest_std.to_frame(name=diet)], axis=1)

    # group plot
    row_colors_groups = pd.DataFrame.from_dict(my_palette, orient='index').rename(columns={0: 'diet'})
    sns.clustermap(all_rest_ave.transpose(), col_cluster=False, row_cluster=False, cmap=cmr.neutral, vmin=0, vmax=1,
                   method='ward',
                   figsize=(3.7, 1), row_colors=row_colors_groups)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_mean.pdf"), dpi=350)
    plt.close()

    # species plot
    # need to order the dat first by 00:00, and then by diet group
    select_diet = cichlid_meta.loc[:, ['six_letter_name_Ronco', 'diet']].rename(columns={'six_letter_name_Ronco': 'species'}).drop_duplicates()

    # invert data
    # aves_ave_rest_inv = abs(aves_ave_rest - 1)

    ordering_df = aves_ave_rest.transpose().reset_index().rename(columns={'index': 'species'})

    ordering_df = ordering_df.merge(loadings.loc[:, ['species', 'pc1', 'pc2']], how='left', on='species')
    ordering_df = ordering_df.merge(select_diet, how='left', on='species')
    ordering_df = ordering_df.merge(diel_guilds, how='left', on='species')

    ordering_df = ordering_df.sort_values(by=['diet', '00:00']).set_index('species')
    row_colors_sp = ordering_df['diet'].map(my_palette)
    row_colors_diel = ordering_df['diel_guild'].map(my_palette_diel)
    ordered_df = ordering_df.iloc[:, 0:48]

    sns.clustermap(ordered_df, col_cluster=False, row_cluster=False, cmap=cmr.neutral_r, vmin=0, vmax=1,
                   figsize=(3.7, 5), row_colors=row_colors_sp, yticklabels=1)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_one_big.pdf"), dpi=350)
    plt.close()

    # combined species and group plot
    species_and_groups = pd.concat([ordered_df, all_rest_ave.transpose()])
    row_colors_sp_and_groups = pd.concat([row_colors_sp, row_colors_groups])
    row_colors_sp_and_groups_and_diel = pd.concat([row_colors_sp_and_groups, row_colors_diel], axis=1)
    sns.clustermap(species_and_groups, col_cluster=False, row_cluster=False, cmap=cmr.neutral_r, vmin=0, vmax=1,
                   figsize=(2, 5.5), row_colors=row_colors_sp_and_groups_and_diel, yticklabels=1)
    plt.savefig(os.path.join(rootdir, "temporal_niche_partitioning_combined_mean.pdf"), dpi=350)
    plt.close()
    return


if __name__ == '__main__':
    rootdir = select_dir_path()

    loadings = pd.read_csv(os.path.join(rootdir, 'pca_loadings.csv'))
    diel_guilds = pd.read_csv(os.path.join(rootdir, 'diel_guilds.csv'))

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

    plot_temporal_niche_one(rootdir, aves_ave_rest, loadings, diel_guilds)

