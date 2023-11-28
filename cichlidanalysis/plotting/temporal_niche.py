import os

import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec

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
        select_data = aves_ave_rest.transpose().loc[select_sp]
        select_data_inv = abs(select_data - 1)
        # rest_sum_norm = select_data.sum(axis=1)/select_data.shape[0]
        non_rest_max = select_data_inv.max(axis=0)
        non_rest_ave = select_data_inv.mean(axis=0)

        row_colors = pd.DataFrame(data=[my_palette[diet]] * len(select_data_inv.index.to_list()),
                                  index=select_data_inv.index.to_list())

        cg = sns.clustermap(select_data_inv, col_cluster=False, cmap='RdPu_r', vmin=0, vmax=1, method='ward',
                            figsize=(3.7, 3.7), row_colors=row_colors)
        cg.ax_row_dendrogram.set_visible(False)  # suppress row dendrogram
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


if __name__ == '__main__':
    rootdir = select_dir_path()

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

    plot_temporal_niche(rootdir, aves_ave_rest)
