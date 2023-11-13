from tkinter.filedialog import askdirectory
from tkinter import *
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
# insipired by https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


def plot_loadings(rootdir, pca, labels, data_input):
    loadings = pd.DataFrame(pca.components_.T, columns=labels, index=data_input.columns)

    for pc_name in labels:
        loadings_sorted = loadings.sort_values(by=pc_name)
        f, ax = plt.subplots(figsize=(15, 5))
        plt.scatter(loadings_sorted.index, loadings_sorted.loc[:, pc_name])
        ax.set_xticklabels(loadings_sorted.index, rotation=90)
        plt.title(pc_name)
        ax.set_ylabel('loading')
        sns.despine(top=True, right=True)
        plt.axhline(0, color='gainsboro')
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "loadings_{}.png".format(pc_name)), dpi=1000)
        plt.close()
    return loadings


def plot_2D_pc_space(rootdir, finalDf, target):
    all_target = finalDf.loc[:, target].unique()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    cmap = matplotlib.cm.get_cmap('nipy_spectral')
    for species_n, species_name in enumerate(all_target):
        indicesToKeep = finalDf[target] == species_name
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
                   color=cmap(species_n / len(all_target)), s=50)
    ax.legend(all_target)
    # ax.scatter(finalDf.loc[:, 'pc1'], finalDf.loc[:, 'pc2'], s=50)
    ax.grid()
    plt.savefig(os.path.join(rootdir, "PCA_points_2D_space_{}.png".format(target)), dpi=1000)
    plt.close()
    return


def plot_2D_pc_space_colour(rootdir, finalDf, target):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    im = ax.scatter(finalDf.loc[:, 'pc1'], finalDf.loc[:, 'pc2'], c=finalDf.loc[:, target], s=50)
    # Add a colorbar
    fig.colorbar(im, ax=ax)

    ax.grid()
    plt.savefig(os.path.join(rootdir, "PCA_points_2D_space_coloured-by-{}.png".format(target)), dpi=1000)
    plt.close()
    return


def plot_2D_pc_space_label_species(rootdir, finalDf, target):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    ax.scatter(finalDf.pc1, finalDf.pc2)
    # Add labels to the points
    for i, label in enumerate(target):
        plt.annotate(label, (finalDf.pc1[i], finalDf.pc2[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # ax.scatter(finalDf.loc[:, 'pc1'], finalDf.loc[:, 'pc2'], s=50)
    ax.grid()
    plt.savefig(os.path.join(rootdir, "PCA_points_2D_space_labelled{}.png".format('species')), dpi=1000)
    plt.close()
    return


def plot_2D_pc_space_orig(rootdir, data_input, finalDf):
    # plot 2D PC space with labeled points
    day = set(np.where(data_input.index.to_series().reset_index(drop=True) < '19:00')[0]) & set(
        np.where(data_input.index.to_series().reset_index(drop=True) >= '07:00')[0])
    six_thirty_am = set(np.where(data_input.index.to_series().reset_index(drop=True) == '06:30')[0])
    seven_am = set(np.where(data_input.index.to_series().reset_index(drop=True) == '07:00')[0])
    seven_pm = set(np.where(data_input.index.to_series().reset_index(drop=True) == '19:00')[0])
    six_thirty_pm = set(np.where(data_input.index.to_series().reset_index(drop=True) == '18:30')[0])
    finalDf['daynight'] = 'night'
    finalDf.loc[day, 'daynight'] = 'day'
    finalDf.loc[six_thirty_am, 'daynight'] = 'six_thirty_am'
    finalDf.loc[six_thirty_pm, 'daynight'] = 'six_thirty_pm'
    finalDf.loc[seven_am, 'daynight'] = 'seven_am'
    finalDf.loc[seven_pm, 'daynight'] = 'seven_pm'

    cmap = matplotlib.cm.get_cmap('twilight_shifted')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    times = finalDf.daynight
    timepoints = times.unique()
    # colors = ['r', 'g', 'b']
    for time_n, time in enumerate(timepoints):
        indicesToKeep = finalDf['daynight'] == time
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
                   c=cmap(time_n / len(timepoints)), s=50)
    ax.legend(timepoints)
    ax.grid()
    plt.savefig(os.path.join(rootdir, "PCA.png"), dpi=1000)
    plt.close()
    return


def plot_variance_explained(rootdir, principalDf, pca):
    f, ax = plt.subplots(figsize=(5, 5))
    cmap = matplotlib.cm.get_cmap('flare')
    x = np.arange(0, principalDf.shape[0])
    for col_n, col in enumerate(principalDf.columns):
        y = principalDf.loc[:, col]
        plt.plot(y, c=cmap(col_n / principalDf.shape[1]), label=col)
    plt.legend()
    plt.savefig(os.path.join(rootdir, "principalDf.png"), dpi=1000)
    plt.close()

    f, ax = plt.subplots(figsize=(5, 5))
    plt.bar(np.arange(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='r', marker='o', linestyle='-')
    plt.ylim([0, 1])
    sns.despine(top=True, right=True)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Cumulative fraction of variance explained')
    plt.savefig(os.path.join(rootdir, "explained_variance_.png"), dpi=1000)
    plt.close()
    return


def plot_factor_loading_matrix(rootdir, loadings, top_pc=3):
    """ Plot the factor loading matrix for top X pcs

    :param rootdir:
    :param loadings:
    :param top_pc:
    :return:
    """
    fig, ax = plt.subplots(figsize=(5, 15))
    sns.heatmap(loadings.iloc[:, :top_pc], annot=True, cmap="seismic")
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "factor_loading_matrix.png"))
    plt.close()

    sns.clustermap(loadings.iloc[:, :top_pc], annot=True, cmap="seismic", figsize=(5, 15), col_cluster=False)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "factor_loading_matrix_clustered.png"))
    plt.close()
    return


def pc_loadings_on_2D(rootdir, principalComponents, coeff, loadings, top_n):
    # sorting loadings
    loadings_i = loadings.reset_index()
    # for pc1 and pc2 find indices that are the top 3 + and -
    ls = []
    ls.extend(loadings_i.sort_values('pc1').iloc[0:top_n, 0].index.values)
    ls.extend(loadings_i.sort_values('pc1').iloc[-top_n:, 0].index.values)
    ls.extend(loadings_i.sort_values('pc2').iloc[0:top_n, 0].index.values)
    ls.extend(loadings_i.sort_values('pc2').iloc[-top_n:, 0].index.values)


    xs = principalComponents[:, 0]
    ys = principalComponents[:, 1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    fig, ax = plt.subplots(figsize=(5, 15))
    plt.scatter(xs * scalex, ys * scaley, color='gainsboro')
    for i in ls:
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, loadings.index[i], color='k', ha='center', va='center',
                 fontsize=5)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.savefig(os.path.join(rootdir, "pc_loadings_on_2D.png"), dpi=1000)
    plt.close()
    return


def plot_reconstruct_pc(rootdir, data_input, pca, mu, pc_n):
    # reconstruct the data with only pc 'n'
    Xhat = np.dot(pca.transform(data_input)[:, pc_n - 1:pc_n], pca.components_[:1, :])
    Xhat += mu
    reconstructed = pd.DataFrame(data=Xhat, columns=data_input.columns)
    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(reconstructed)
    plt.savefig(os.path.join(rootdir, "reconstruction_from_pc{}.png".format(pc_n)), dpi=1000)
    plt.close()
    return


def plot_3D_pc_space(rootdir, finalDf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = finalDf['pc1']
    y = finalDf['pc2']
    z = finalDf['pc3']

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 2')
    ax.set_title('3 component PCA')

    ax.plot(x, z, color='dimgrey', zdir='y', zs=20, markersize=0.5, marker='o', linestyle='None')
    ax.plot(y, z, color='dimgrey', zdir='x', zs=-20, markersize=0.5, marker='o', linestyle='None')
    ax.plot(x, y, color='dimgrey', zdir='z', zs=-20, markersize=0.5, marker='o', linestyle='None')

    ax.scatter(x, y, z, color='forestgreen')

    ax.set_xlim([-20, 25])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])

    plt.savefig(os.path.join(rootdir, "PCA_points_3D_space.png"), dpi=1000)
    plt.close()
    return
