import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression


def run_linear_reg(x, y):
    """  Runs linear regression on x and y  varialbes. Will reshape x as needed (so  you  can add in pandas series)

    :param x: behavioural feature, pandas series
    :param y: ecological feature, pandas series
    :return: model and r_sq (R2 value for model fit)
    """
    x = x.to_numpy()
    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return model, r_sq


def plt_lin_reg(rootdir, x, y, model, r_sq, label='', name_x=False, name_y=False, labels=False):
    """ Plot the scatter of two variables and add in the linear regression model, pearson's correlation and R2

    :param name_y: name for y axis
    :param name_x: name for x axis
    :param x: behavioural feature, pandas series
    :param y: ecological feature, pandas series
    :param model: run_linear_reg model
    :param r_sq: R2 value for model fit
    :param label: label to add to save name
    :return: saves plot
    """
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    fig = plt.figure(figsize=(1.5, 1.5))
    ax = sns.scatterplot(x, y, s=3)
    # x_intercept = (model.intercept_/model.coef_)[0]*-1
    # plt.plot([0, x_intercept], [model.intercept_, 0], color='k')
    y_pred = model.predict(np.reshape(x.to_numpy(), (-1, 1)))
    plt.plot(x, y_pred, color='k', linewidth=1)
    ax.set(xlim=(min(x)-0.05, max(x)+0.05), ylim=(min(y)-0.05, max(y)+0.05))

    r = np.corrcoef(x, y)
    ax.text(0.85, 0.90, s="$R^2$={}".format(np.round(r_sq, 2)), va='top', ha='center', transform=ax.transAxes)
    ax.text(0.85, 0.96, s="$R$={}".format(np.round(r[0, 1], 2)), va='top', ha='center', transform=ax.transAxes)
    fig.tight_layout()
    # ax.set_ylabel('Day - night activity')
    # ax.set_xlabel('Peak fraction')
    if name_x:
        ax.set_xlabel(name_x)
    if name_y:
        ax.set_ylabel(name_y)
    if labels:
        n = 0
        for (i, j) in zip(x, y):
            plt.text(i, j, x.index[n])
            n = n+1

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.savefig(os.path.join(rootdir, "feature_correlation_plot_{0}_vs_{1}_{2}.pdf".format(x.name, y.name, label)), dpi=350)
    plt.close()

    # # residuals plot
    # fig = plt.figure(figsize=(5, 5))
    # ax = sns.scatterplot(x, y-y_pred)
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Residuals')
    # plt.axhline(y=0, color='k')
    # fig.tight_layout()
    return


def feature_correlations(rootdir, feature_v_mean, fv_eco_sp_ave):
    feature_v_mean_i = feature_v_mean.set_index('six_letter_name_Ronco')
    for behav in ['total_rest', 'day_night_dif', 'size_female']:
        for col in ['body_PC1', 'body_PC2', 'LPJ_PC1', 'LPJ_PC2', 'oral_PC1', 'oral_PC2', 'd15N', 'd13C', 'size_male',
                    'size_female']:
            non_nan_rows = feature_v_mean_i[feature_v_mean_i[behav].isna() == False].index & fv_eco_sp_ave[fv_eco_sp_ave[col].isna() == False].index
            model, r_sq = run_linear_reg(feature_v_mean_i.loc[non_nan_rows, behav], fv_eco_sp_ave.loc[non_nan_rows, col])
            plt_lin_reg(rootdir, feature_v_mean_i.loc[non_nan_rows, behav], fv_eco_sp_ave.loc[non_nan_rows, col], model, r_sq)

    # correlating day/night vs crepuscularity
    model, r_sq = run_linear_reg(feature_v_mean_i.peak, abs(feature_v_mean_i.day_night_dif))
    plt_lin_reg(rootdir, feature_v_mean_i.peak, abs(feature_v_mean_i.day_night_dif), model, r_sq)

    model, r_sq = run_linear_reg(feature_v_mean_i.peak_amplitude, abs(feature_v_mean_i.day_night_dif))
    plt_lin_reg(rootdir, feature_v_mean_i.peak_amplitude, abs(feature_v_mean_i.day_night_dif), model, r_sq)

    model, r_sq = run_linear_reg(feature_v_mean_i.peak, feature_v_mean_i.day_night_dif)
    plt_lin_reg(rootdir, feature_v_mean_i.peak, feature_v_mean_i.day_night_dif, model, r_sq)

    model, r_sq = run_linear_reg(feature_v_mean_i.peak_amplitude, feature_v_mean_i.day_night_dif)
    plt_lin_reg(rootdir, feature_v_mean_i.peak_amplitude, feature_v_mean_i.day_night_dif, model, r_sq)

    model, r_sq = run_linear_reg(feature_v_mean_i.total_rest, feature_v_mean_i.day_night_dif)
    plt_lin_reg(rootdir, feature_v_mean_i.total_rest, feature_v_mean_i.day_night_dif, model, r_sq)

    model, r_sq = run_linear_reg(feature_v_mean_i.total_rest, feature_v_mean_i.peak)
    plt_lin_reg(rootdir, feature_v_mean_i.total_rest, feature_v_mean_i.peak, model, r_sq)

    model, r_sq = run_linear_reg(feature_v_mean_i.total_rest, feature_v_mean_i.fish_length_mm)
    plt_lin_reg(rootdir, feature_v_mean_i.total_rest, feature_v_mean_i.fish_length_mm, model, r_sq)
    plt.close('all')

    # fig = plt.figure(figsize=(5, 5))
    # sns.regplot(data=df, x='total_rest', y='night-day_dif_rest')
    #
    # fig = plt.figure(figsize=(5, 5))
    # sns.scatterplot(data=df, x='d15N', y='d13C', hue='total_rest')
    return
