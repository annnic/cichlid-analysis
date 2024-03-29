import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.spatial as spatial
from scipy import stats
import cmasher as cmr

from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg


def plot_ecospace_vs_temporal_guilds(rootdir, feature_v_eco, ronco_data, fv_eco_sp_ave, diel_guilds):
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    my_palette_diel = {'Diurnal': '#CED926', 'Nocturnal': '#40A9BF', 'Crepuscular': '#26D97A', 'Cathemeral': '#737F8C'}

    # pelagic and trophic levels (ecospace) vs temporal guilds
    fig = plt.figure(figsize=(1.5, 1.5))
    feature_v_eco_all_sp_ave = feature_v_eco.groupby(by='six_letter_name_Ronco').mean()
    ronco_data_ave = ronco_data.groupby(by='sp').mean()
    plt.scatter(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=8, alpha=0.7,
                         edgecolors='none')
    ax = plt.gca()
    for key in diel_guilds.diel_guild.unique():
        # find the species which are in diel group
        guild_species = set(diel_guilds.loc[diel_guilds.diel_guild == key, 'species'].unique())
        overlap_species = list(guild_species & set(fv_eco_sp_ave.index.to_list()))
        points = fv_eco_sp_ave.loc[overlap_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1], color=my_palette_diel[key], s=8, alpha=0.7, edgecolors='none')
        hull = spatial.ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], color=my_palette_diel[key])
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_temporal-guilds.pdf"), dpi=350)
    plt.close()
    return


def plot_d15N_d13C_diet_guilds(rootdir, feature_v_eco, fv_eco_sp_ave, ronco_data):
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    guilds = feature_v_eco.diet.unique()
    diet_col_dic = {'Zooplanktivore': 'sandybrown', 'Algivore': 'mediumseagreen', 'Invertivore': 'tomato',
                    'Piscivore': 'steelblue'}
    fig = plt.figure(figsize=(1.5, 1.5))
    ronco_data_ave = ronco_data.groupby(by='sp').mean()
    plt.scatter(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=8, alpha=0.7,
                         edgecolors='none')
    ax = plt.gca()
    for key in guilds:
        # find the species which are in the diet guild
        guild_species = set(feature_v_eco.loc[feature_v_eco.diet == key, 'six_letter_name_Ronco'].unique())
        points = fv_eco_sp_ave.loc[guild_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1], color=diet_col_dic[key], s=8, alpha=0.7, edgecolors='none')
        if key in ['Zooplanktivore', 'Algivore', 'Invertivore', 'Piscivore']:
            hull = spatial.ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], color=diet_col_dic[key])
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_diet-guilds.pdf"), dpi=350)
    plt.close()
    return


def plot_diet_guilds_hist(rootdir, feature_v_eco, dic_simple, diel_patterns):

    first = True
    for key in dic_simple:
        cluster_sp = diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species'].to_list()
        new_df = feature_v_eco.loc[feature_v_eco.six_letter_name_Ronco.isin(cluster_sp), ['six_letter_name_Ronco',
                                                                    'diet']].drop_duplicates().diet.value_counts()
        new_df = new_df.reset_index()
        new_df['daytime'] = key
        if first:
            df_group = new_df
            first = False
        else:
            df_group = pd.concat([df_group, new_df])
    df_group = df_group.rename(columns={'diet': 'species_n', 'index': 'diet'}).reset_index(drop=True)

    colors = ['sandybrown', 'tomato', 'mediumseagreen', 'steelblue']
    customPalette = sns.set_palette(sns.color_palette(colors))
    fig = plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="daytime", y="species_n", hue="diet", data=df_group, palette=customPalette)
    ax.set(xlabel=None)
    ax.set(ylabel="# of species")
    plt.savefig(os.path.join(rootdir, "diet-guilds_hist.png"), dpi=1200)
    plt.close()
    return


def plot_total_rest_vs_diet_significance(rootdir, feature_v_eco):
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    feature_v_eco_species = feature_v_eco.loc[:, ['diet', 'habitat', 'six_letter_name_Ronco', 'total_rest', 'rest_mean_night',
                                'rest_mean_day', 'fish_length_mm', 'night-day_dif_rest', 'fish_n', 'species_true',
                                'species_our_names', 'species_six']].drop_duplicates().reset_index(drop=True)
    colors = ['mediumseagreen', 'sandybrown', 'tomato', 'steelblue']
    diet_order = ['Algivore', 'Zooplanktivore', 'Invertivore', 'Piscivore']

    stats_array = np.zeros([len(diet_order), len(diet_order)])
    for diet_1_n, diet_1 in enumerate(diet_order):
        for diet_2_n, diet_2 in enumerate(diet_order):
            _, stats_array[diet_1_n, diet_2_n] = stats.ttest_ind(feature_v_eco_species.loc[feature_v_eco_species.diet
                                                                                           == diet_1,'total_rest'],
                                                                 feature_v_eco_species.loc[feature_v_eco_species.diet ==
                                                                                           diet_2, 'total_rest'])
    fig = plt.figure(figsize=(2, 2))
    ax = sns.boxplot(data=feature_v_eco_species, x='diet', y='total_rest', dodge=False, showfliers=False, order=diet_order)
    ax = sns.swarmplot(data=feature_v_eco_species, x='diet', y='total_rest', color=".2", size=2, order=diet_order)
    ax.set(xlabel='Diet Guild', ylabel='Average total rest per day')
    ax.set(ylim=(0, 24))
    plt.xticks(rotation='45', ha="right")

    # statistical annotation
    if not np.max(stats_array < 0.05):
        y, h, col = 22, len(diet_order), 'k'
        for diet_i_n, diet_i in enumerate(diet_order):
            plt.text(diet_i_n, y, "ns", ha='center', va='bottom', color=col)

    sns.despine(top=True, right=True)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "total_rest_vs_diet_significance.pdf"), dpi=350)
    plt.close()
    return


def plot_ecospace_vs_feature(rootdir, ronco_data, loadings, fv_eco_sp_ave, pc='pc1', cmap_n='coolwarm'):
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    fig = plt.figure(figsize=(2, 1.5))
    ronco_data_ave = ronco_data.groupby(by='sp').mean()
    plt.scatter(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=8, alpha=0.7,
                edgecolors='none')
    overlap = set(fv_eco_sp_ave.index).intersection(set(loadings.species))
    points = fv_eco_sp_ave.loc[overlap, ['d13C', 'd15N']].rename_axis('species')
    points = pd.merge(points, loadings, on='species')

    cmap = plt.get_cmap(cmap_n)
    scatter = plt.scatter(points['d13C'], points['d15N'], c=points[pc], cmap=cmap, s=8, alpha=0.7, edgecolors='none')

    ax = plt.gca()
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    cbar = plt.colorbar(scatter, label='{} loading'.format(pc), shrink=0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_{}.pdf".format(pc)), dpi=350)
    plt.close()
    return


def plot_ecospace_vs_feature_rest(rootdir, ronco_data, cichild_corr_data, fv_eco_sp_ave, feature='total_rest', cmap_n='coolwarm'):
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    fig = plt.figure(figsize=(2, 1.5))
    ronco_data_ave = ronco_data.groupby(by='sp').mean()
    plt.scatter(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=8, alpha=0.7,
                edgecolors='none')
    overlap = set(fv_eco_sp_ave.index).intersection(set(cichild_corr_data.species))
    points = fv_eco_sp_ave.loc[overlap, ['d13C', 'd15N']].rename_axis('species')
    points = pd.merge(points, cichild_corr_data.loc[:, [feature, 'species']], on='species')

    cmap = plt.get_cmap(cmap_n)
    scatter = plt.scatter(points['d13C'], points['d15N'], c=points[feature], cmap=cmap, s=8, alpha=0.7, edgecolors='none')

    ax = plt.gca()
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    cbar = plt.colorbar(scatter, label='{} loading'.format(feature), shrink=0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_{}.pdf".format(feature)), dpi=350)
    plt.close()
    return