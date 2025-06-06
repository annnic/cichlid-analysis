import os
from tkinter import *
from tkinter.filedialog import askdirectory
import math
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg, plt_lin_reg_rest_figure


def read_tps(file_path):
    specimens = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        specimen = {}
        landmarks = []
        for line in lines:
            line = line.strip()
            if line.startswith("LM="):
                if specimen:
                    specimen['landmarks'] = landmarks
                    specimens.append(specimen)
                specimen = {'landmarks': []}
                landmarks = []
                specimen['num_landmarks'] = int(line.split('=')[1])
            elif line.startswith("ID="):
                specimen['id'] = line.split('=')[1]
            else:
                # Parse landmark coordinates
                coords = list(map(float, line.split()))
                landmarks.append(coords)
        if specimen:
            specimen['landmarks'] = landmarks
            specimens.append(specimen)
    return specimens


def read_spd_percentiles(rootdir):
    csv_files = glob.glob(os.path.join(os.path.join(rootdir, "_spd_percentiles"), '*.csv'))

    df_list = [pd.read_csv(file) for file in csv_files]
    spd_percent = pd.concat(df_list, ignore_index=True)
    first_column = spd_percent.columns[0]
    spd_percent.rename(columns={first_column: 'FishID'}, inplace=True)
    return spd_percent


def read_rest_bl(rootdir):
    csv_files = glob.glob(os.path.join(os.path.join(rootdir, "rest_body_lengths"), '*_total_rest_bl.csv'))

    df_list = [pd.read_csv(file) for file in csv_files]
    rest_bl = pd.concat(df_list, ignore_index=True)
    rest_bl.drop(columns=rest_bl.columns[0], axis=1, inplace=True)
    rest_bl.rename(columns={'ID': 'FishID'}, inplace=True)
    return rest_bl


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    # load data
    loadings = pd.read_csv(os.path.join(rootdir, "pca_loadings.csv"), sep=',')
    cichlid_data = pd.read_csv(os.path.join(rootdir, "cichild_pc-loadings_eco-morph_rest_full.csv"), sep=',')
    table_1 = pd.read_csv(os.path.join(rootdir, "table_1.csv"), sep=',')
    diel_guilds = pd.read_csv(os.path.join(rootdir, "diel_guilds.csv"), sep=',')
    tps_data = read_tps(os.path.join(rootdir, "06_landmark_data_body_shape.tps"))
    explore_data = pd.read_csv(os.path.join(rootdir, "exploratoryBehaviorMedians.txt"), sep='\t')
    explore_data = explore_data.drop(columns=['species_id'])
    explore_data = explore_data.rename(columns={'species_abb': 'species'})
    spd_percent = read_spd_percentiles(rootdir)
    rest_bl = read_rest_bl(rootdir)

    # get the species for each specimen/entry of the tps file
    for specimen_n, specimen in enumerate(tps_data):
        if specimen_n == 0:
            specimen_species = pd.DataFrame({'species': [specimen['id'][0:-4].split('_')[1]]})
        else:
            specimen_species_i = pd.DataFrame({'species': [specimen['id'][0:-4].split('_')[1]]})
            specimen_species = pd.concat([specimen_species, specimen_species_i], ignore_index=True)

    first_species = True
    for _, species in enumerate(loadings.species):
        specimens_for_species = specimen_species.loc[specimen_species.species == species]

        if not specimens_for_species.empty:
            first = True
            for specimen_n in specimens_for_species.index:
                specimen_data = tps_data[specimen_n]
                # note the -1 compared to the point numbers due to zero indexing
                eye_west = specimen_data['landmarks'][16]
                eye_east = specimen_data['landmarks'][18]
                eye_size_h = math.sqrt((eye_west[0] - eye_east[0]) ** 2 + (eye_west[1] - eye_east[1]) ** 2)

                eye_north = specimen_data['landmarks'][17]
                eye_south = specimen_data['landmarks'][19]
                eye_size_v = math.sqrt((eye_north[0] - eye_south[0]) ** 2 + (eye_north[1] - eye_south[1]) ** 2)

                nose = specimen_data['landmarks'][0]
                tail = specimen_data['landmarks'][6]
                standard_len = math.sqrt((nose[0] - tail[0]) ** 2 + (nose[1] - tail[1]) ** 2)

                # get body outline (1, 3, 4, 7, 8, 11)
                outline = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15])
                points = np.ones((len(outline), 2))
                points[:] = np.nan
                for i, element in enumerate(outline):
                    points[i, :] = specimen_data['landmarks'][element-1]
                # Compute the convex hull of body
                hull = ConvexHull(points)
                body_area = hull.area

                # plotting the points and hull
                input_points = np.array(specimen_data['landmarks'][0:-1])
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
                for ax in (ax1, ax2):
                    ax.plot(input_points[:, 0], input_points[:, 1], '.', color='k')
                    if ax == ax1:
                        ax.set_title('Given points')
                    else:
                        ax.set_title('Convex hull')
                        for simplex in hull.simplices:
                            ax.plot(points[simplex, 0], points[simplex, 1], 'c')
                        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1,
                                markersize=6)

                        # for point in input_points[16:20, :]:
                        ax.plot(input_points[16:20, 0], input_points[16:20, 1], 'o', mec='g', color='none', lw=1,
                                markersize=6)
                    ax.set_xticks(range(10))

                    ax.set_yticks(range(10))
                    ax.set_xlim(0, 10)
                    ax.set_ylim(-12, 0)
                plt.savefig(os.path.join(rootdir, "convex_hull_{}.png".format(species)))
                plt.close('all')

                if first:
                    species_measures = pd.DataFrame({'species': [specimen_data['id'][0:-4].split('_')[1]], 'ID': specimen_data['id'],
                                                     'eye_size_h': eye_size_h, 'eye_size_v': eye_size_v, 'standard_len':
                                                         standard_len, 'body_area': body_area})
                    first = False
                else:
                    species_measures_i = pd.DataFrame({'species': [specimen_data['id'][0:-4].split('_')[1]], 'ID': specimen_data['id'],
                                                     'eye_size_h': eye_size_h, 'eye_size_v': eye_size_v, 'standard_len':
                                                         standard_len, 'body_area': body_area})
                    species_measures = pd.concat([species_measures, species_measures_i], ignore_index=True)

            if first_species:
                all_species_measures = species_measures
                first_species = False
            else:
                all_species_measures = pd.concat([all_species_measures, species_measures], ignore_index=True)
        else:
            print('Not including {}'.format(species))

    all_species_measures['eye_v_by_sl'] = all_species_measures.loc[:, 'eye_size_v'] / all_species_measures.loc[:,'standard_len']
    all_species_measures['eye_h_by_sl'] = all_species_measures.loc[:, 'eye_size_h'] / all_species_measures.loc[:,'standard_len']
    all_species_measures['eye_h_by_ba'] = all_species_measures.loc[:, 'eye_size_h'] / all_species_measures.loc[:,'body_area']

    all_species_measures_mean = all_species_measures.groupby('species').mean().reset_index()
    all_species_measures_mean.to_csv(os.path.join(rootdir, 'body_measurements.csv'), sep=',', index=False,
                                     encoding='utf-8')

    combined_eye_loadings = pd.merge(cichlid_data, all_species_measures_mean, on='species')
    combined_eye_loadings = combined_eye_loadings.set_index('species')

    combined_eye_loadings.reset_index().to_csv(os.path.join(rootdir, 'body_measurements.csv'), sep=',', index=False,
                                     encoding='utf-8')


    # combine exploration data and loadings
    combined_explore_loadings = pd.merge(cichlid_data, explore_data, on='species')
    combined_explore_loadings = combined_explore_loadings.set_index('species')

    # pc1 vs day_night_dif
    model, r_sq = run_linear_reg(combined_eye_loadings.pc1, combined_eye_loadings.eye_v_by_sl)
    plt_lin_reg(rootdir, combined_eye_loadings.pc1, combined_eye_loadings.eye_v_by_sl, model, r_sq,
                name_x='pc1 loading', name_y='relative eye size', labels=True, figsize=(4, 4))

    model, r_sq = run_linear_reg(combined_eye_loadings.pc1, combined_eye_loadings.eye_h_by_sl)
    plt_lin_reg(rootdir, combined_eye_loadings.pc1, combined_eye_loadings.eye_h_by_sl, model, r_sq,
                name_x='pc1 loading', name_y='relative eye size h', labels=True, figsize=(4, 4))

    model, r_sq = run_linear_reg(combined_eye_loadings.eye_v_by_sl, combined_eye_loadings.eye_h_by_sl)
    plt_lin_reg(rootdir, combined_eye_loadings.eye_v_by_sl, combined_eye_loadings.eye_h_by_sl, model, r_sq,
                name_x='relative eye size v', name_y='relative eye size h', labels=True, figsize=(4, 4))

    model, r_sq = run_linear_reg(combined_eye_loadings.pc1, combined_eye_loadings.eye_h_by_ba)
    plt_lin_reg(rootdir, combined_eye_loadings.pc1, combined_eye_loadings.eye_h_by_ba, model, r_sq,
                name_x='pc1 loading', name_y='relative eye size ba', labels=True, figsize=(4, 4))

    model, r_sq = run_linear_reg(combined_eye_loadings.standard_len, combined_eye_loadings.body_area)
    plt_lin_reg(rootdir, combined_eye_loadings.standard_len, combined_eye_loadings.body_area, model, r_sq,
                name_x='standard_len', name_y='body_area', labels=True, figsize=(4, 4))


    model, r_sq = run_linear_reg(combined_eye_loadings.standard_len, combined_eye_loadings.total_rest)
    plt_lin_reg(rootdir, combined_eye_loadings.standard_len, combined_eye_loadings.total_rest, model, r_sq,
                name_x='standard_len', name_y='total_rest', labels=True, figsize=(4, 4))

    model, r_sq = run_linear_reg(combined_eye_loadings.body_area, combined_eye_loadings.total_rest)
    plt_lin_reg(rootdir, combined_eye_loadings.body_area, combined_eye_loadings.total_rest, model, r_sq,
                name_x='body_area', name_y='total_rest', labels=True, figsize=(4, 4))

    # exploration
    model, r_sq = run_linear_reg(combined_explore_loadings.pc1, combined_explore_loadings.median_exploration)
    plt_lin_reg(rootdir, combined_explore_loadings.pc1, combined_explore_loadings.median_exploration, model, r_sq,
                name_x='pc1 loading', name_y='exploration', labels=True, figsize=(4, 4))

    model, r_sq = run_linear_reg(combined_explore_loadings.pc2, combined_explore_loadings.median_exploration)
    plt_lin_reg(rootdir, combined_explore_loadings.pc2, combined_explore_loadings.median_exploration, model, r_sq,
                name_x='pc2 loading', name_y='exploration', labels=True, figsize=(4, 4))


    # speed vs body size
    # note that FISH20200907_c6_r1_Neolamprologus-gracilis_sm has nan for speed, drop
    spd_percent = spd_percent.dropna()

    model, r_sq = run_linear_reg(spd_percent.fish_length_mm, spd_percent.iloc[:, 1])
    plt_lin_reg(rootdir, spd_percent.fish_length_mm, spd_percent.iloc[:, 1], model, r_sq,
                name_x='fish_length_mm', name_y='50 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent.fish_length_mm, spd_percent.iloc[:, 2])
    plt_lin_reg(rootdir, spd_percent.fish_length_mm, spd_percent.iloc[:, 2], model, r_sq,
                name_x='fish_length_mm', name_y='90 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent.fish_length_mm, spd_percent.iloc[:, 3])
    plt_lin_reg(rootdir, spd_percent.fish_length_mm, spd_percent.iloc[:, 3], model, r_sq,
                name_x='fish_length_mm', name_y='95 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent.fish_length_mm, spd_percent.iloc[:, 4])
    plt_lin_reg(rootdir, spd_percent.fish_length_mm, spd_percent.iloc[:, 4], model, r_sq,
                name_x='fish_length_mm', name_y='98 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent.fish_length_mm, spd_percent.iloc[:, 5])
    plt_lin_reg(rootdir, spd_percent.fish_length_mm, spd_percent.iloc[:, 5], model, r_sq,
                name_x='fish_length_mm', name_y='99 percentile', labels=False, figsize=(4, 4))

    spd_percent_sp = spd_percent.groupby('species').mean()
    model, r_sq = run_linear_reg(spd_percent_sp.fish_length_mm, spd_percent_sp.iloc[:, 5])
    plt_lin_reg(rootdir, spd_percent_sp.fish_length_mm, spd_percent_sp.iloc[:, 5], model, r_sq,
                name_x='fish_length_mm_species', name_y='995 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent_sp.fish_length_mm, spd_percent_sp.iloc[:, 5])
    plt_lin_reg(rootdir, spd_percent_sp.fish_length_mm, spd_percent_sp.iloc[:, 5], model, r_sq,
                name_x='fish_length_mm_species', name_y='995 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent_sp.fish_length_mm, spd_percent_sp.iloc[:, 4])
    plt_lin_reg(rootdir, spd_percent_sp.fish_length_mm, spd_percent_sp.iloc[:, 4], model, r_sq,
                name_x='fish_length_mm_species', name_y='99 percentile', labels=False, figsize=(4, 4))


    spd_percent['spd_norm'] = spd_percent.iloc[:, 5] / spd_percent.loc[:, 'fish_length_mm']
    model, r_sq = run_linear_reg(spd_percent.spd_norm, spd_percent.iloc[:, 5])
    plt_lin_reg(rootdir, spd_percent.spd_norm, spd_percent.iloc[:, 5], model, r_sq,
                name_x='spd_norm', name_y='99 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent.spd_norm, spd_percent.fish_length_mm)
    plt_lin_reg(rootdir, spd_percent.spd_norm, spd_percent.fish_length_mm, model, r_sq,
                name_x='spd_norm', name_y='fish_length_mm', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent.iloc[:, 5], spd_percent.fish_length_mm)
    plt_lin_reg(rootdir, spd_percent.iloc[:, 5], spd_percent.fish_length_mm, model, r_sq,
                name_x='99 percentile', name_y='fish_length_mm', labels=False, figsize=(4, 4))


    spd_percent_sp['spd_norm'] = spd_percent_sp.iloc[:, 5] / spd_percent_sp.loc[:, 'fish_length_mm']
    model, r_sq = run_linear_reg(spd_percent_sp.spd_norm, spd_percent_sp.iloc[:, 5])
    plt_lin_reg(rootdir, spd_percent_sp.spd_norm, spd_percent_sp.iloc[:, 5], model, r_sq,
                name_x='spd_norm', name_y='995 percentile', labels=False, figsize=(4, 4))

    model, r_sq = run_linear_reg(spd_percent_sp.spd_norm, spd_percent_sp.fish_length_mm)
    plt_lin_reg(rootdir, spd_percent_sp.spd_norm, spd_percent_sp.fish_length_mm, model, r_sq,
                name_x='spd_norm', name_y='fish_length_mm', labels=False, figsize=(4, 4))


    # rest with 15mm/s or 0.25bl thresholds
    rest_bl = rest_bl.dropna()
    model, r_sq = run_linear_reg(rest_bl.total_rest, rest_bl.total_rest_bl)
    plt_lin_reg(rootdir, rest_bl.total_rest, rest_bl.total_rest_bl, model, r_sq,
                name_x='total_rest mm/s', name_y='total_rest 0.25 body lengths', labels=False, figsize=(4, 4))

    plt_lin_reg_rest_figure(rootdir, rest_bl.total_rest, rest_bl.total_rest_bl, model, r_sq,
                name_x='total_rest mm/s', name_y='total_rest 0.25 body lengths', labels=False, figsize=(3, 3))

    # grouped by species
    rest_bl_sp = rest_bl.groupby('species').mean()
    model, r_sq = run_linear_reg(rest_bl_sp.total_rest, rest_bl_sp.total_rest_bl)
    plt_lin_reg(rootdir, rest_bl_sp.total_rest, rest_bl_sp.total_rest_bl, model, r_sq,
                name_x='total_rest mm/s', name_y='total_rest 0.25 body lengths', labels=True, figsize=(4, 4))
    plt.close('all')

    # rest definition figure extendend figure 2e

