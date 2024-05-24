import os
from tkinter import *
from tkinter.filedialog import askdirectory
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py_tps import TPSFile

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
    # voucher_key = pd.read_csv(os.path.join(rootdir, "01_specimen_voucher_key.csv"), sep=',')
    # body_shape_tps =TPSFile.read_file(os.path.join(rootdir, "06_landmark_data_body_shape.tps"))

    tps_data = read_tps(os.path.join(rootdir, "06_landmark_data_body_shape.tps"))

    # for specimen in tps_data:
    #     print("Specimen ID:", specimen['id'])
    #     print("Number of Landmarks:", specimen['num_landmarks'])
    #     print("Landmarks:", specimen['landmarks'])

    # get the species for each specimen/entry of the tps file
    for specimen_n, specimen in enumerate(tps_data):
        if specimen_n == 0:
            specimen_species = pd.DataFrame({'species': [specimen['id'].split('_')[1]]})
        else:
            specimen_species_i = pd.DataFrame({'species': [specimen['id'].split('_')[1]]})
            specimen_species = pd.concat([specimen_species, specimen_species_i], ignore_index=True)

    first_species = True
    for species_n, species in enumerate(loadings.species):
        specimens_for_species = specimen_species.loc[specimen_species.species == species]

        if not specimens_for_species.empty:
            first = True
            for specimen_n in specimens_for_species.index:
                specimen_data = tps_data[specimen_n]
                eye_west = specimen_data['landmarks'][17]
                eye_east = specimen_data['landmarks'][19]
                eye_size_h = math.sqrt((eye_west[0] - eye_east[0]) ** 2 + (eye_west[1] - eye_east[1]) ** 2)

                eye_north = specimen_data['landmarks'][18]
                eye_south = specimen_data['landmarks'][20]
                eye_size_v = math.sqrt((eye_north[0] - eye_south[0]) ** 2 + (eye_north[1] - eye_south[1]) ** 2)

                nose = specimen_data['landmarks'][1]
                tail = specimen_data['landmarks'][7]
                standard_len = math.sqrt((nose[0] - tail[0]) ** 2 + (nose[1] - tail[1]) ** 2)

                if first:
                    species_measures = pd.DataFrame({'species': [specimen_data['id'].split('_')[1]], 'ID': specimen_data['id'],
                                                     'eye_size_h': eye_size_h, 'eye_size_v': eye_size_v, 'standard_len':
                                                         standard_len})
                    first = False
                else:
                    species_measures_i = pd.DataFrame({'species': [specimen_data['id'].split('_')[1]], 'ID': specimen_data['id'],
                                                     'eye_size_h': eye_size_h, 'eye_size_v': eye_size_v, 'standard_len':
                                                         standard_len})
                    species_measures = pd.concat([species_measures, species_measures_i], ignore_index=True)

            if first_species:
                all_species_measures = species_measures
                first_species = False
            else:
                all_species_measures = pd.concat([all_species_measures, species_measures], ignore_index=True)

    all_species_measures['eye_v_by_sl'] = all_species_measures.loc[:, 'eye_size_v'] / all_species_measures.loc[:,'standard_len']
    all_species_measures['eye_h_by_sl'] = all_species_measures.loc[:, 'eye_size_h'] / all_species_measures.loc[:,'standard_len']

    all_species_measures_mean = all_species_measures.groupby('species').mean().reset_index()

    combined_eye_loadings = pd.merge(loadings, all_species_measures_mean, on='species')
    combined_eye_loadings = combined_eye_loadings.set_index('species')
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