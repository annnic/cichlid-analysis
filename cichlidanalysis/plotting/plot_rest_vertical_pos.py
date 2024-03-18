import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import glob
from scipy.stats import ttest_rel

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.io.io_ecological_measures import get_meta_paths


def plot_rest_vs_vp(rootdir, rest_vp_sp):
    """ total rest ordered by mean

    :param rootdir:
    :param feature_v:
    :return:
    """
    # font sizes
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})
    sns.set_context(rc={"lines.linewidth": 0.5})
    hue_colors = {'0': 'lightcoral', '1': 'lightsteelblue'}

    all_species = rest_vp_sp.six_letter_name_Ronco.unique()
    first = True
    for species in all_species:
        df_subset_non_rest = rest_vp_sp.loc[(rest_vp_sp.six_letter_name_Ronco == species) & (rest_vp_sp.rest == 0),
                                        ['vertical_pos']]
        df_subset_rest = rest_vp_sp.loc[(rest_vp_sp.six_letter_name_Ronco == species) & (rest_vp_sp.rest == 1),
                                        ['vertical_pos']]

        t_statistic, p_value = ttest_rel(df_subset_non_rest, df_subset_rest)

        if first:
            pvals = dict({species: p_value[0]})
            first = False
        else:
            pvals[species] = p_value[0]
    rest_vp_sp['rest'] = rest_vp_sp['rest'].astype(str)

    fig = plt.figure(figsize=(7, 1.5))
    ax = sns.boxplot(data=rest_vp_sp, x='six_letter_name_Ronco', y='vertical_pos',
                     showfliers=False, linewidth=0.5, hue='rest', palette=hue_colors,
                     order=rest_vp_sp.loc[rest_vp_sp.rest == '1', :].groupby('six_letter_name_Ronco').mean().sort_values("vertical_pos").index.to_list())
    ax = sns.swarmplot(data=rest_vp_sp, x='six_letter_name_Ronco', y='vertical_pos', size=1, hue='rest', color='black',
                       dodge=True, order=rest_vp_sp.loc[rest_vp_sp.rest == '1', :].groupby('six_letter_name_Ronco').mean().sort_values("vertical_pos").index.to_list(), linewidth=0.5)
    ax.set(ylabel='Vertical position', xlabel='Species')
    ax.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.axhline(12, ls='--', color='k', lw=0.5)
    sns.despine(top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)

    num_sp = len(all_species)
    # bonferrroni correction
    alpha_corrected = 0.05 / num_sp
    pvals_sig = pd.DataFrame.from_dict([pvals]).transpose()
    pvals_sig.rename(columns={pvals_sig.columns[0]: "pval"}, inplace=True)
    pvals_sig['pval_sig'] = 'ns'
    pvals_sig.loc[pvals_sig.pval < alpha_corrected, 'pval_sig'] = '*'
    for i, species in enumerate(all_species):
        x_pos = i
        y_pos = 1
        plt.text(x_pos, y_pos,  pvals_sig.loc[pvals_sig.index == species, 'pval_sig'].values[0], ha='center',
                 va='bottom', size=SMALLEST_SIZE)

    plt.savefig(os.path.join(rootdir, "vertical_position_by_rest_horizontal.pdf"), dpi=350)
    plt.close()
    return


def load_rest_vp(folder, suffix="rest_vp_*.csv"):
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

    data = data.drop(columns=["Unnamed: 0"])

    print("All rest_vp_species.csv files loaded")
    return data


if __name__ == '__main__':
    rootdir = select_dir_path()

    # load all "rest_vp_{species}.csv"
    rest_vp = load_rest_vp(rootdir, suffix="rest_vp_*.csv")

    # need the ronco 6 letter names
    _, cichlid_meta_path = get_meta_paths()
    sp_metrics = pd.read_csv(cichlid_meta_path)

    # need to remove the '-' in the species names
    rest_vp['species'] = rest_vp.loc[:, 'FishID'].str.split('_').str[3].str.replace('-', ' ')

    rest_vp = rest_vp.rename(columns={"species": "species_our_names"})
    rest_vp_sp = rest_vp.merge(sp_metrics, on='species_our_names', how='left')

    plot_rest_vs_vp(rootdir, rest_vp_sp)




