import os

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.analysis.linear_regression import run_linear_reg, plt_lin_reg


if __name__ == '__main__':
    rootdir = select_dir_path()

    loadings = pd.read_csv(os.path.join(rootdir, 'pca_loadings.csv'))
    loadings = loadings.set_index('species')
    corr_vals_long_daily = pd.read_csv(os.path.join(rootdir, 'corr_vals_long_daily.csv'))
    corr_vals_long_weekly = pd.read_csv(os.path.join(rootdir, 'corr_vals_long_weekly.csv'))

    mean_corrs_daily = pd.pivot_table(corr_vals_long_daily, index='species', values='corr_coef', aggfunc='mean')
    mean_corrs_daily = mean_corrs_daily.rename(columns={"corr_coef": "corr_coef_daily"})
    mean_corrs_weekly = pd.pivot_table(corr_vals_long_weekly, index='species', values='corr_coef', aggfunc='mean')
    mean_corrs_weekly = mean_corrs_weekly.rename(columns={"corr_coef": "corr_coef_weekly"})

    # correlations
    data1 = loadings.pc1
    data2 = mean_corrs_daily.corr_coef_daily
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)

    data1 = loadings.pc1
    data2 = mean_corrs_weekly.corr_coef_weekly
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)

    data1 = loadings.pc2
    data2 = mean_corrs_daily.corr_coef_daily
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)

    data1 = loadings.pc2
    data2 = mean_corrs_weekly.corr_coef_weekly
    model, r_sq = run_linear_reg(data1, data2)
    plt_lin_reg(rootdir, data1, data2, model, r_sq)