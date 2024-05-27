import os
from tkinter import *
from tkinter.filedialog import askdirectory
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg


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


if __name__ == '__main__':
    # Allows user to select top directory and load all als files here
    root = Tk()
    rootdir = askdirectory(parent=root)
    root.destroy()

    # load data
    loadings = pd.read_csv(os.path.join(rootdir, "pca_loadings.csv"), sep=',')
    table_1 = pd.read_csv(os.path.join(rootdir, "table_1.csv"), sep=',')
    diel_guilds = pd.read_csv(os.path.join(rootdir, "diel_guilds.csv"), sep=',')
    tps_data = read_tps(os.path.join(rootdir, "06_landmark_data_body_shape.tps"))

    # get the species for each specimen/entry of the tps file
    for specimen_n, specimen in enumerate(tps_data):
        if specimen_n == 0:
            specimen_species = pd.DataFrame({'species': [specimen['id'][0:-4].split('_')[1]]})
        else:
            specimen_species_i = pd.DataFrame({'species': [specimen['id'][0:-4].split('_')[1]]})
            specimen_species = pd.concat([specimen_species, specimen_species_i], ignore_index=True)

    first_species = True
    for species_n, species in enumerate(loadings.species):
        specimens_for_species = specimen_species.loc[specimen_species.species == species]

        if not specimens_for_species.empty:
            first = True
            for specimen_n in specimens_for_species.index:
                specimen_data = tps_data[specimen_n]
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

                ## plotting the points and hull
                # input_points = np.array(specimen_data['landmarks'][0:-1])
                # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
                # for ax in (ax1, ax2):
                #     ax.plot(input_points[:, 0], input_points[:, 1], '.', color='k')
                #     if ax == ax1:
                #         ax.set_title('Given points')
                #     else:
                #         ax.set_title('Convex hull')
                #         for simplex in hull.simplices:
                #             ax.plot(points[simplex, 0], points[simplex, 1], 'c')
                #         ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1,
                #                 markersize=10)
                #
                #         # for point in input_points[16:20, :]:
                #         ax.plot(input_points[16:20, 0], input_points[16:20, 1], 'o', mec='g', color='none', lw=1,
                #                 markersize=10)
                #     ax.set_xticks(range(10))
                #
                #     ax.set_yticks(range(10))
                # # plt.savefig(os.path.join(rootdir, "convex_hull_2.png"))
                # plt.close('all')

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

    combined_eye_loadings = pd.merge(loadings, all_species_measures_mean, on='species')
    combined_eye_loadings = combined_eye_loadings.set_index('species')

    combined_eye_loadings.reset_index().to_csv(os.path.join(rootdir, 'body_measurements.csv'), sep=',', index=False,
                                     encoding='utf-8')

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
    plt.close('all')
