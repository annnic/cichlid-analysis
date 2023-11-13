import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial as spatial
from scipy import stats

from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg


def plot_ecospace_vs_temporal_guilds(rootdir, feature_v_eco, ronco_data, diel_patterns, dic_simple, col_dic_simple, fv_eco_sp_ave):
    # pelagic and trophic levels (ecospace) vs temporal guilds
    fig = plt.figure(figsize=(3, 3))
    feature_v_eco_all_sp_ave = feature_v_eco.groupby(by='six_letter_name_Ronco').mean()
    ronco_data_ave = ronco_data.groupby(by='sp').mean()
    ax = sns.scatterplot(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=12)
    for key in dic_simple:
        # find the species which are in diel group
        overlap_species = list(
            set(diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species'].to_list()) &
            set(fv_eco_sp_ave.index.to_list()))
        points = fv_eco_sp_ave.loc[overlap_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1], color=col_dic_simple[key], s=12)
        if key in ['diurnal', 'nocturnal', 'crepuscular']:
            hull = spatial.ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], color=col_dic_simple[key], alpha=0.4)
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_temporal-guilds.png"), dpi=1200)
    plt.close()
    return


def plot_ecospace_vs_temporal_guilds_density(rootdir, ronco_data, diel_patterns, dic_simple, col_dic_simple, fv_eco_sp_ave):

    ronco_data_ave = ronco_data.groupby(by='sp').mean()

    col_list = []
    for ordered_col in fv_eco_sp_ave.cluster_pattern.unique():
        col_list.append(col_dic_simple[ordered_col])

    fig = plt.figure(figsize=(3, 3))
    ax = sns.displot(fv_eco_sp_ave, x="d13C", y="d15N", hue="cluster_pattern", kind="kde", levels=2, palette=col_list)
    plt.scatter(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=12)
    for key in dic_simple:
        # find the species which are in diel group
        overlap_species = list(
            set(diel_patterns.loc[diel_patterns.cluster.isin(dic_simple[key]), 'species'].to_list()) &
            set(fv_eco_sp_ave.index.to_list()))
        points = fv_eco_sp_ave.loc[overlap_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1], color=col_dic_simple[key], s=12)
    plt.xlabel('$\delta^{13} C$')
    plt.ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_temporal-guilds_density.png"), dpi=1200)
    plt.close()





def plot_d15N_d13C_diet_guilds(rootdir, feature_v_eco, fv_eco_sp_ave, ronco_data):
    guilds = feature_v_eco.diet.unique()
    diet_col_dic = {'Zooplanktivore': 'sandybrown', 'Algivore': 'mediumseagreen', 'Invertivore': 'tomato',
                    'Piscivore': 'steelblue'}
    fig = plt.figure(figsize=(3, 3))
    ronco_data_ave = ronco_data.groupby(by='sp').mean()
    ax = sns.scatterplot(ronco_data_ave.loc[:, 'd13C'], ronco_data_ave.loc[:, 'd15N'], color='silver', s=12)
    for key in guilds:
        # find the species which are in the diet guild
        guild_species = set(feature_v_eco.loc[feature_v_eco.diet == key, 'six_letter_name_Ronco'].unique())
        points = fv_eco_sp_ave.loc[guild_species, ['d13C', 'd15N']]
        points = points.to_numpy()
        plt.scatter(points[:, 0], points[:, 1], color=diet_col_dic[key], s=12)
        if key in ['Zooplanktivore', 'Algivore', 'Invertivore', 'Piscivore']:
            hull = spatial.ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], color=diet_col_dic[key])
    ax.set_xlabel('$\delta^{13} C$')
    ax.set_ylabel('$\delta^{15} N$')
    sns.despine(top=True, right=True)
    fig.tight_layout()
    plt.savefig(os.path.join(rootdir, "d15N_d13C_diet-guilds.png"), dpi=1200)
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
    feature_v_eco_species = feature_v_eco.loc[:, ['diet', 'habitat', 'six_letter_name_Ronco', 'total_rest', 'rest_mean_night',
                                'rest_mean_day', 'fish_length_mm', 'night-day_dif_rest', 'fish_n', 'species_true',
                                'species_our_names', 'species_six']].drop_duplicates().reset_index(drop=True)
    colors = ['mediumseagreen', 'sandybrown', 'tomato', 'steelblue']
    diet_order = ['Algivore', 'Zooplanktivore', 'Invertivore', 'Piscivore']
    customPalette = sns.set_palette(sns.color_palette(colors))

    stats_array = np.zeros([len(diet_order), len(diet_order)])
    for diet_1_n, diet_1 in enumerate(diet_order):
        for diet_2_n, diet_2 in enumerate(diet_order):
            _, stats_array[diet_1_n, diet_2_n] = stats.ttest_ind(feature_v_eco_species.loc[feature_v_eco_species.diet
                                                                                           == diet_1,'total_rest'],
                                                                 feature_v_eco_species.loc[feature_v_eco_species.diet ==
                                                                                           diet_2, 'total_rest'])
    fig = plt.figure(figsize=(3, 3))
    ax = sns.boxplot(data=feature_v_eco_species, x='diet', y='total_rest', dodge=False, showfliers=False, order=diet_order)
    ax = sns.swarmplot(data=feature_v_eco_species, x='diet', y='total_rest', color=".2", size=4, order=diet_order)
    ax.set(xlabel='Diet Guild', ylabel='Average total rest per day')
    ax.set(ylim=(0, 24))
    plt.xticks(rotation='45', ha="right")

    # statistical annotation
    if not np.max(stats_array < 0.05):
        y, h, col = 22, len(diet_order), 'k'
        for diet_i_n, diet_i in enumerate(diet_order):
            plt.text(diet_i_n, y, "ns", ha='center', va='bottom', color=col)

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "total_rest_vs_diet_significance.png"))
    plt.close()
    return
