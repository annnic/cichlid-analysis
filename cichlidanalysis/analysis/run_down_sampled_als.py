from tkinter.filedialog import askdirectory
from tkinter import *
import warnings
import os
import copy
import statistics

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import hsv
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from cichlidanalysis.io.meta import add_meta_from_name, extract_meta
from cichlidanalysis.io.tracks import load_ds_als_files
from cichlidanalysis.utils.timings import load_timings
from cichlidanalysis.utils.species_names import shorten_sp_name, six_letter_sp_name
from cichlidanalysis.utils.species_metrics import add_metrics, tribe_cols
from cichlidanalysis.plotting.speed_plots import plot_spd_30min_combined
from cichlidanalysis.analysis.processing import feature_daily, species_feature_fish_daily_ave, \
    fish_tracks_add_day_twilight_night, add_day_number_fish_tracks
from cichlidanalysis.analysis.diel_pattern import replace_crep_peaks, make_fish_peaks_df, diel_pattern_ttest_individ_ds

# debug pycharm problem
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_n_colors(n):
    """
    from: https://stackoverflow.com/questions/33246065/convert-categorical-variable-to-color-with-matplotlib
    :param n:
    :return:
    """
    return [cool(float(i) / n) for i in range(n)]


def fish_weekly_corr(fish_tracks_ds, feature, link_method):
    """

    :param fish_tracks_ds:
    :param feature:
    :param link_method:
    :return:
    """
    species = fish_tracks_ds['species'].unique()
    first = True

    for species_i in species:
        print(species_i)
        fish_tracks_ds_sp = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, ['FishID', 'ts', feature]]
        fish_tracks_ds_sp = fish_tracks_ds_sp.pivot(columns='FishID', values=feature, index='ts')
        individ_corr = fish_tracks_ds_sp.corr()

        mask = np.ones(individ_corr.shape, dtype='bool')
        mask[np.triu_indices(len(individ_corr))] = False
        corr_val_f = individ_corr.values[mask]

        if first:
            corr_vals = pd.DataFrame(corr_val_f, columns=[six_letter_sp_name(species_i)[0]])
            first = False
        else:
            corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_val_f, columns=[six_letter_sp_name(species_i)[0]])], axis=1)

        X = individ_corr.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method=link_method)
        ind = sch.fcluster(L, 0.5*d.max(), 'distance')
        cols = [individ_corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
        individ_corr = individ_corr[cols]
        individ_corr = individ_corr.reindex(cols)

        fish_sex = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, ['FishID', 'sex']].drop_duplicates()
        fish_sex = list(fish_sex.sex)
        fish_sex_clus = [fish_sex[i] for i in list((np.argsort(ind)))]

        f, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(data=individ_corr, vmin=-1, vmax=1, xticklabels=fish_sex_clus, yticklabels=fish_sex_clus,
                         cmap='seismic', ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "{0}_corr_by_30min_{1}_{2}_{3}.png".format(species_i.replace(' ', '-'),
                                                                                     feature, dt.date.today(),
                                                                                     link_method)))
        plt.close()

    corr_vals_long = pd.melt(corr_vals, var_name='species_six', value_name='corr_coef')

    f, ax = plt.subplots(figsize=(3, 5))
    sns.boxplot(data=corr_vals_long, x='corr_coef', y='species_six', ax=ax, fliersize=0)
    sns.stripplot(data=corr_vals_long, x='corr_coef', y='species_six', color=".2", ax=ax, size=3)
    ax.set(xlabel='Correlation', ylabel='Species')
    ax.set(xlim=(-1, 1))
    ax = plt.axvline(0, ls='--', color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "fish_corr_coefs_{0}_{1}.png".format(feature,  dt.date.today())))
    plt.close()

    return corr_vals


def fish_daily_corr(averages_feature, feature, species_name, link_method='single'):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :param species_name:
    :param link_method:
    :return:
    """

    individ_corr = averages_feature.corr()

    ax = sns.clustermap(individ_corr, figsize=(7, 5), method=link_method, metric='euclidean', vmin=-1, vmax=1,
                        cmap='RdBu_r', xticklabels=False, yticklabels=False)
    ax.fig.suptitle(feature)
    plt.savefig(os.path.join(rootdir, "fish_of_{0}_corr_by_30min_{1}_{2}_{3}.png".format(species_name, feature, dt.date.today(), link_method)))
    plt.close()



def species_daily_corr(averages_feature, feature, link_method='single'):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """

    individ_corr = averages_feature.corr()

    ax = sns.clustermap(individ_corr, figsize=(7, 5), method=link_method, metric='euclidean', vmin=-1, vmax=1, cmap='RdBu_r', yticklabels=True)
    ax.fig.suptitle(feature)
    plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}_{1}_{2}.png".format(feature, dt.date.today(), link_method)))
    plt.close()


def week_corr(fish_tracks_ds, feature):
    """ Plots corr matrix of clustered species by given feature

    :param averages_feature:
    :param feature:
    :return:
    """
    species = fish_tracks_ds['species'].unique()

    for species_i in species:

        fishes = fish_tracks_ds.loc[fish_tracks_ds.species == species_i, 'FishID'].unique()
        first = True

        for fish in fishes:
            print(fish)
            fish_tracks_ds_day = fish_tracks_ds.loc[fish_tracks_ds.FishID == fish, ['day', 'time_of_day_dt', feature]]
            fish_tracks_ds_day = fish_tracks_ds_day.pivot(columns='day', values=feature, index='time_of_day_dt')
            individ_corr = fish_tracks_ds_day.corr()

            mask = np.ones(individ_corr.shape, dtype='bool')
            mask[np.triu_indices(len(individ_corr))] = False
            corr_val_f = individ_corr.values[mask]
            if first:
                corr_vals = pd.DataFrame(corr_val_f, columns=[fish])
                first = False
            else:
                corr_vals = pd.concat([corr_vals, pd.DataFrame(corr_val_f, columns=[fish])], axis=1)

            X = individ_corr.values
            d = sch.distance.pdist(X)
            L = sch.linkage(d, method='single')
            ind = sch.fcluster(L, 0.5*d.max(), 'distance')
            cols = [individ_corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
            individ_corr = individ_corr[cols]
            individ_corr = individ_corr.reindex(cols)

            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(individ_corr, vmin=-1, vmax=1, cmap='bwr')
            ax.set_title(fish)
            plt.tight_layout()
            plt.savefig(os.path.join(rootdir, "species_corr_by_30min_{0}_{1}.png".format(feature, dt.date.today())))
            plt.close()

        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.catplot(data=corr_vals, kind="swarm", vmin=-1, vmax=1)
        ax.set_title(species_i)


def add_day(fish_df):
    """ Adds day number to the fish  dataframe, by using the timestamp (ts) column

    :param fish_df:
    :return:
    """
    # add new column with day number (starting from 1)
    fish_df['day'] = fish_df.ts.apply(lambda row: int(str(row)[8:10]) - 1)
    print("added night and day column")
    return fish_df


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    root.withdraw()
    root.update()
    rootdir = askdirectory(parent=root)
    root.destroy()

    fish_tracks_ds = load_ds_als_files(rootdir, "*als_30m.csv")
    fish_tracks_ds = fish_tracks_ds.reset_index(drop=True)
    fish_tracks_ds['time_of_day_dt'] = fish_tracks_ds.ts.apply(lambda row: int(str(row)[11:16][:-3]) * 60 + int(str(row)[11:16][-2:]))
    fish_tracks_ds.loc[fish_tracks_ds.species == 'Aaltolamprologus calvus', 'species'] = 'Altolamprologus calvus'

    # get each fish ID and all species
    fish_IDs = fish_tracks_ds['FishID'].unique()
    species = fish_tracks_ds['species'].unique()

    # reorganising
    species_short = shorten_sp_name(species)
    species_sixes = six_letter_sp_name(species)

    tribe_col = tribe_cols()

    metrics_path = '/Users/annikanichols/Desktop/cichlid_species_database.xlsx'
    sp_metrics = add_metrics(species_sixes, metrics_path)

    # get timings
    fps, tv_ns, tv_sec, tv_24h_sec, num_days, tv_s_type, change_times_s, change_times_ns, change_times_h, day_ns, day_s,\
        change_times_d, change_times_m = load_timings(fish_tracks_ds[fish_tracks_ds.FishID == fish_IDs[0]].shape[0])
    change_times_unit = [7*2, 7.5*2, 18.5*2, 19*2]
    change_times_datetime = [dt.datetime.strptime("1970-1-2 07:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 07:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 18:30:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-2 19:00:00", '%Y-%m-%d %H:%M:%S'),
                             dt.datetime.strptime("1970-1-3 00:00:00", '%Y-%m-%d %H:%M:%S')]
    day_unit = dt.datetime.strptime("1970:1", "%Y:%d")

    feature, ymax, span_max, ylabeling = 'vertical_pos', 1, 0.8, 'Vertical position'
    averages_vp, date_time_obj_vp, sp_vp_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                            ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'speed_mm', 95, 80, 'Speed mm/s'
    averages_spd, _, sp_spd_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'rest', 1, 0.8, 'Rest'
    averages_rest, _, sp_rest_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)
    feature, ymax, span_max, ylabeling = 'movement', 1, 0.8, 'Movement'
    averages_move, _, sp_move_combined = plot_spd_30min_combined(fish_tracks_ds, feature, ymax, span_max,
                                                                              ylabeling, change_times_datetime, rootdir)

    # generate
    aves_ave_spd = feature_daily(averages_spd)
    aves_ave_vp = feature_daily(averages_vp)
    aves_ave_rest = feature_daily(averages_rest)
    aves_ave_move = feature_daily(averages_move)

    aves_ave_spd.columns = species_sixes
    aves_ave_vp.columns = species_sixes
    aves_ave_rest.columns = species_sixes
    aves_ave_move.columns = species_sixes

    row_cols = []
    for i in sp_metrics.tribe:
        row_cols.append(tribe_col[i])

    row_cols_2 = pd.DataFrame(row_cols, index=[aves_ave_spd.columns.tolist()]).apply(tuple, axis=1)
    row_cols_1 = pd.DataFrame(row_cols).apply(tuple, axis=1)
    # ax = sns.clustermap(aves_ave_spd.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
    #                     metric='correlation',
    #                     row_colors=row_cols_2)

    ax = sns.clustermap(aves_ave_spd.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Speed mm/s")
    ax = sns.clustermap(aves_ave_spd.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Speed mm/s")
    plt.close()
    ax = sns.clustermap(aves_ave_vp.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Vertical position")
    plt.close()
    ax = sns.clustermap(aves_ave_rest.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Rest")
    ax = sns.clustermap(aves_ave_rest.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1, yticklabels=True)
    ax.fig.suptitle("Rest")
    plt.close()
    ax = sns.clustermap(aves_ave_move.T, figsize=(7, 5), col_cluster=False, method='single', metric='correlation',
                        yticklabels=True)
    ax.fig.suptitle("Movement")
    ax = sns.clustermap(aves_ave_move.T.reset_index(drop=True), figsize=(7, 5), col_cluster=False, method='single',
                        metric='correlation', row_colors=row_cols_1)
    ax.fig.suptitle("Movement")
    plt.close()

    # clustering of daily average of individuals, massive clustermap!
    features = ['speed_mm', 'rest',  'movement']
    for feature in features:
        first = True
        for species_name in species:
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)

            if first:
                all_fish_daily_ave_feature = fish_daily_ave_feature
                first = False
            else:
                all_fish_daily_ave_feature = pd.concat([all_fish_daily_ave_feature, fish_daily_ave_feature], axis=1)

        col_species = add_meta_from_name(all_fish_daily_ave_feature.columns, 'species').T
        species_code = col_species.species.astype('category').cat.codes

        num_categories = len(set(col_species.species))
        colors = [hsv(float(i) / num_categories) for i in species_code]

        sns.clustermap(all_fish_daily_ave_feature, row_cluster=False, col_colors=colors, xticklabels=col_species.species)
        plt.savefig(os.path.join(rootdir, "all_fish_daily_clustered_30min_{0}_{1}.png".format(feature, dt.date.today())))

# ## correlations ##
    fish_tracks_ds = add_day(fish_tracks_ds)
    # correlations for days across week for an individual
    # week_corr(fish_tracks_ds, 'rest')

    features = ['speed_mm', 'rest']
    for feature in features:
        for species_name in species:
            # correlations for individuals across daily average
            fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)
            fish_daily_corr(fish_daily_ave_feature, feature, species_name)

        # correlations for individuals across week
        _ = fish_weekly_corr(fish_tracks_ds, feature, 'single')

    # correlations for species
    species_daily_corr(aves_ave_spd, 'speed_mm', 'single')
    species_daily_corr(aves_ave_rest, 'rest', 'single')


# better crepuscular
from scipy.signal import find_peaks

border_top = np.ones(48)
border_bottom = np.ones(48)
border_bottom[6*2:8*2] = 0
border_bottom[18*2:20*2] = 0
# features = ['speed_mm', 'rest', 'movement']
# for feature in features:
#     first = True
for species_name in species:
    fish_daily_ave_feature = species_feature_fish_daily_ave(fish_tracks_ds, species_name, feature)
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(fish_daily_ave_feature.T, cmap="Greys")
    for i in np.arange(0, len(fish_daily_ave_feature.columns)):
        x = fish_daily_ave_feature.iloc[:, i]
        peaks, _ = find_peaks(x, distance=4, prominence=0.15, height=(border_bottom, border_top))
        plt.plot(x)
        plt.plot(peaks, x[peaks],  "o", color="r")
        plt.plot(border_bottom)
        plt.plot(x.reset_index().index[peaks].values, (np.ones(len(peaks))*i)+0.5,   "o", color="r")
        plt.title(species_name)
        plt.show()


border_bottom = np.concatenate((border_bottom, border_bottom, border_bottom, border_bottom, border_bottom,
                                border_bottom, border_bottom))
border_top = np.ones(48*7)
for species_name in species:
    fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
        columns='FishID', values=feature, index='ts')
    fig = plt.figure(figsize=(10, 5))
    sns.heatmap(fish_feature.T, cmap="Greys")
    for i in np.arange(0, len(fish_feature.columns)):
        x = fish_feature.iloc[:, i]
        peaks, peak_prop = find_peaks(x, distance=4, prominence=0.15, height=(border_bottom[0:x.shape[0]], border_top[0:x.shape[0]]))

        np.around(peak_prop['peak_heights'], 2)

        # plt.plot(x)
        # plt.plot(peaks, x[peaks],  "o", color="r")
        # plt.plot(border_bottom)
        plt.plot(x.reset_index().index[peaks].values, (np.ones(len(peaks))*i)+0.5,   "o", color="r")
        plt.title(species_name)
        plt.show()

# need: peak height, peak location, dawn/dusk, max day/nightfor  that day, if  peak missing, find most common peak
# location and use  the  value of that  bin. Find amplitude of peaks
border_top = np.ones(48)
border_bottom = np.ones(48)
dawn_border_bottom  = copy.copy(border_bottom)
dawn_border_bottom[6*2:(8*2)+1] = 0
dusk_border_bottom  = copy.copy(border_bottom)
dusk_border_bottom[18*2:(20*2)+1] = 0

border = np.zeros(48)
day_border = copy.copy(border)
day_border[8*2:18*2] = 1
night_border = copy.copy(border)
night_border[0:6*2] = 1
night_border[20*2:24*2] = 1


# check what happens when that crepuscaulr period is missing!
first_all= True
for species_name in species:
    fish_feature = fish_tracks_ds.loc[fish_tracks_ds.species == species_name, ['ts', 'FishID', feature]].pivot(
        columns='FishID', values=feature, index='ts')
    first = True
    for i in np.arange(0, len(fish_feature.columns)):
        epoques = np.arange(0, 48*7.5, 48).astype(int)
        fish_peaks_dawn = np.zeros([4, int(np.floor(fish_feature.iloc[:, i].reset_index().shape[0]/48))])
        fish_peaks_dusk = np.zeros([4, int(np.floor(fish_feature.iloc[:, i].reset_index().shape[0] / 48))])
        for j in np.arange(0, int(np.ceil(fish_feature.shape[0]/48))):
            x = fish_feature.iloc[epoques[j]:epoques[j+1], i]
            if x.size == 48:
                #dawn peak
                dawn_peak, dawn_peak_prop = find_peaks(x, distance=4, prominence=0.15,height=
                (dawn_border_bottom[0:x.shape[0]], border_top[0:x.shape[0]]))

                # duskpeak
                dusk_peak, dusk_peak_prop = find_peaks(x, distance=4, prominence=0.15, height=
                (dusk_border_bottom[0:x.shape[0]],border_top[0:x.shape[0]]))

                # fig = plt.figure(figsize=(10, 5))
                # plt.plot(x)

                if dawn_peak.size != 0:
                    fish_peaks_dawn[0, j] = dawn_peak[0]
                    fish_peaks_dawn[1, j] = dawn_peak[0] + epoques[j]
                    fish_peaks_dawn[2, j] = np.round(dawn_peak_prop['peak_heights'][0], 2)
                    # plt.plot(dawn_peak[0], x[int(dawn_peak[0])], "o", color="r")

                if dusk_peak.size != 0:
                    fish_peaks_dusk[0, j] = dusk_peak[0]
                    fish_peaks_dusk[1, j] = dusk_peak[0] + epoques[j]
                    fish_peaks_dusk[2, j] = np.round(dusk_peak_prop['peak_heights'][0], 2)
                    # plt.plot(dusk_peak[0], x[int(dusk_peak[0])], "o", color="r")

                # plt.plot(dawn_border_bottom)
                # plt.plot(dusk_border_bottom)

                # day mean
                day_mean = np.round(x[(day_border).astype(int) == 1].mean(), 2)
                # night mean
                night_mean = np.round(x[(night_border).astype(int) == 1].mean(), 2)

                fish_peaks_dawn[3, j] = fish_peaks_dawn[2, j] - np.max([day_mean, night_mean])
                fish_peaks_dusk[3, j] = fish_peaks_dusk[2, j] - np.max([day_mean, night_mean])

                # keep track  of if day or night max
        fish_peaks_dawn = replace_crep_peaks(fish_peaks_dawn, fish_feature, i, epoques)
        fish_peaks_dusk = replace_crep_peaks(fish_peaks_dusk, fish_feature, i, epoques)

        fish_peaks_df_dawn = make_fish_peaks_df(fish_peaks_dawn, fish_feature.columns[i])
        fish_peaks_df_dusk = make_fish_peaks_df(fish_peaks_dusk, fish_feature.columns[i])

        fish_peaks_df_dawn['twilight'] = 'dawn'
        fish_peaks_df_dusk['twilight'] = 'dusk'
        fish_peaks_df = pd.concat([fish_peaks_df_dawn, fish_peaks_df_dusk], axis=0)

        if first:
            species_peaks_df = fish_peaks_df
            first = False
        else:
            species_peaks_df = pd.concat([species_peaks_df, fish_peaks_df], axis=0)
    species_peaks_df['species'] = species_name
    if first_all:
        all_peaks_df = species_peaks_df
        first_all = False
    else:
        all_peaks_df = pd.concat([all_peaks_df, species_peaks_df], axis=0)

all_peaks_df = all_peaks_df.reset_index(drop=True)
all_peaks_df['peak'] = (all_peaks_df.peak_loc !=0)*1

# fig = plt.figure(figsize=(10, 5))
# sns.swarmplot(x='twilight', y='peak_amplitude', data=all_peaks_df, hue='peak')

fig = plt.figure(figsize=(10, 5))
sns.swarmplot(x='species', y='peak_amplitude', data=all_peaks_df, hue='peak')


# average for each fish for dawn and dusk for 'peak_amplitude', peaks/(peaks+nonpeaks)
periods = ['dawn', 'dusk']
first_all =  True
for species_name in species:
    first  = True
    for period in periods:
        feature_i = all_peaks_df[(all_peaks_df['species']==species_name) & (all_peaks_df['twilight']==period)]
        [['peak_amplitude', 'FishID', 'crep_num', 'peak']]
        sp_average_peak_amp = feature_i.groupby('FishID').mean().peak_amplitude.reset_index()
        sp_average_peak = feature_i.groupby('FishID').mean().peak.reset_index()
        sp_average_peak_data = pd.concat([sp_average_peak_amp, sp_average_peak], axis=1)
        sp_average_peak_data['twilight'] = period
        if first:
            sp_feature_combined = sp_average_peak_data
            first = False
        else:
            sp_feature_combined = pd.concat([sp_feature_combined, sp_average_peak_data], axis=0)
    sp_feature_combined['species'] = species_name

    if first_all:
        all_feature_combined = sp_feature_combined
        first_all = False
    else:
        all_feature_combined = pd.concat([all_feature_combined, sp_feature_combined], axis=0)
all_feature_combined = all_feature_combined.reset_index(drop=True)

fig = plt.figure(figsize=(10, 5))
sns.swarmplot(x='species', y='peak_amplitude', data=all_feature_combined, hue='twilight', dodge=True)


# # calculate ave and stdv
# average = sp_feature.mean(axis=1)
# averages[species_n, :] = average[0:303]
#
#
#
#     fig = plt.figure(figsize=(10, 5))
#     plt.hist(species_peaks_df.peak_amplitude)
#
#     fig = plt.figure(figsize=(10, 5))
#     plt.hist(species_peaks_df_dusk.loc[species_peaks_df_dusk.peak_loc == 0, 'peak_amplitude'])
#
#         x = fish_feature.iloc[:, i]
#         fig = plt.figure(figsize=(10, 5))
#         plt.plot(x)
#         plt.plot(x.reset_index().index[fish_peaks[0, :].astype(int)].values, fish_peaks[1, :],   "o", color="r")
#         plt.title(species_name)
#         plt.show()

# 	1. Find peaks in daily average of Individuals and  species
# 	2. Find peaks across week
# 	3. Find amplitude of peaks
# For non-peaks -  take the most common peak bin
#
# feature_i.loc[feature_i.peak_loc > 0].groupby('FishID').mean().peak_loc
# feature_i.loc[feature_i.peak_loc > 0].groupby('FishID').peak_loc.agg(pd.Series.mode)
#
# x = fish_feature.iloc[:, i]
# plt.plot(fish_peaks[0, :], x[(fish_peaks[0, :]).astype(int)],  "x", color="k")

fish_tracks_ds = fish_tracks_add_day_twilight_night(fish_tracks_ds)
fish_tracks_ds = add_day_number_fish_tracks(fish_tracks_ds)
fish_diel_patterns = diel_pattern_ttest_individ_ds(fish_tracks_ds, feature='speed_mm')

# add back 'species'
fishes = fish_tracks_ds.FishID.unique()
fish_diel_patterns['species_six'] = 'blank'
for fish in fishes:
    fish_diel_patterns.loc[fish_diel_patterns['FishID'] == fish, 'species_six'] = six_letter_sp_name(extract_meta(fish)['species'])

# define species diel pattern
states = ['nocturnal', 'diurnal', ]
fish_diel_patterns['species_diel_pattern'] = 'undefined'
for species_name in species_sixes:
    for state in states:
        if ((fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'diel_pattern'] == state)*1).mean() > 0.5:
            fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'] = state
    print("{} is {}".format(species_name, fish_diel_patterns.loc[fish_diel_patterns.species_six == species_name, 'species_diel_pattern'].unique()))


# row colours
row_cols = []
subset = fish_diel_patterns.loc[:, ['species_six', 'species_diel_pattern']].drop_duplicates(subset = ["species_six"])
for index, row in subset.iterrows():
    if row['species_diel_pattern'] == 'diurnal':
        # row_cols.append(sns.color_palette()[1])
        row_cols.append((255/255, 224/255, 179/255))
    elif row['species_diel_pattern'] == 'nocturnal':
        # row_cols.append(sns.color_palette()[0])
        row_cols.append((153/255, 204/255, 255/255))
    elif row['species_diel_pattern'] == 'undefined':
        # row_cols.append(sns.color_palette()[2])
        row_cols.append((179/255, 230/255, 179/255))

clrs = [(sns.color_palette()[2]), sns.color_palette()[0], sns.color_palette()[1]]

f, ax = plt.subplots(figsize=(5, 10))
sns.boxplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', palette=row_cols,  ax=ax, fliersize=0) #, color="gainsboro"
sns.stripplot(data=fish_diel_patterns, x='day_night_dif', y='species_six', hue='diel_pattern', ax=ax, size=4, palette=clrs)
ax.set(xlabel='Day mean - night mean', ylabel='Species')
# ax.set(xlim=(-1, 1))
ax = plt.axvline(0, ls='--', color='k')
plt.tight_layout()


# def day_night_ratio_individ_30min(fish_tracks_ds, feature='movement'):
#     """ Find the day/night ratio of non-rest for the daily average rest trace of each individual fish
#
#     :param fish_tracks_ds:
#     :return:
#     """
#     # all individuals
#     night = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'n', [feature, 'FishID']].groupby('FishID').mean()
#     day = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'd', [feature, 'FishID']].groupby('FishID').mean()
#
#     day_night_ratio = np.abs(1 - day) / np.abs(1 - night)
#     day_night_ratio = day_night_ratio.rename(columns={feature: "ratio"})
#     day_night_ratio = day_night_ratio.reset_index()
#
#     ax = sns.boxplot(data=day_night_ratio, y='FishID', x='ratio')
#     ax = sns.swarmplot(data=day_night_ratio, y='FishID', x='ratio', color=".2")
#     ax = plt.axvline(1, ls='--', color='k')
#     plt.xlabel('Day/night ratio')
#     plt.xscale('log')
#     plt.tight_layout()
#
#     return day_night_ratio


feature_dist_day = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'd', [feature, 'FishID', 'day_n']].groupby(['FishID', 'day_n']).mean().reset_index()
feature_dist_night = fish_tracks_ds.loc[fish_tracks_ds.daytime == 'n', [feature, 'FishID', 'day_n']].groupby(['FishID', 'day_n']).mean().reset_index()
feature_dist_day = feature_dist_day.rename(columns={feature: "day"})
feature_dist_night = feature_dist_night.rename(columns={feature: "night"})
feature_dist = pd.merge(feature_dist_day, feature_dist_night, how='inner', on=["FishID", "day_n"])
feature_dist['ratio'] = feature_dist.day/feature_dist.night

daily_ratio_mean = feature_dist.loc[:, ['FishID', 'ratio']].groupby(['FishID']).mean().rename(columns={'ratio': "average"})
daily_ratio_std = feature_dist.loc[:, ['FishID', 'ratio']].groupby(['FishID']).std().rename(columns={'ratio': "stdev"})
daily_ratio = pd.merge(daily_ratio_mean, daily_ratio_std, how='inner', on=["FishID"])

daily_ratio['corrected_low'] = daily_ratio.average - daily_ratio.stdev
daily_ratio['corrected_high'] = daily_ratio.average + daily_ratio.stdev

g = sns.catplot(x="ratio", y="FishID",  data=feature_dist)
ax = plt.axvline(1, ls='--', color='k')

plt.imshow(pd.concat([daily_ratio['corrected_low']*1>1,daily_ratio['corrected_high']*1<1], axis=1))
plt.yticks(np.arange(0, len(daily_ratio)), daily_ratio.index)
