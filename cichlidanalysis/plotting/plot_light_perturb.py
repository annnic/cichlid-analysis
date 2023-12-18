import os

import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind
from scipy import stats

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.analysis.run_binned_als import load_bin_als_files
from cichlidanalysis.utils.timings import load_timings_14_8
from cichlidanalysis.plotting.speed_plots import plot_ridge_plots
from cichlidanalysis.plotting.speed_plots import plot_speed_30m_mstd_figure, plot_speed_30m_mstd_figure_light_perturb
from cichlidanalysis.plotting.daily_plots import daily_ave_spd_figure_timed_perturb, \
    daily_ave_spd_figure_timed_perturb_darkdark


def plot_ld_dd_stripplot(rootdir, spd_aves_dn, custom_palette_2, SMALLEST_SIZE, species_f, ttest_control, ttest_dd):
    plt.figure(figsize=(1.5, 1))
    ax = sns.stripplot(data=spd_aves_dn, x='condition', y='speed_mm', hue='condition', s=2,
                       palette=custom_palette_2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0, 80])
    plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)
    plt.title(species_f, fontsize=SMALLEST_SIZE)

    # Decrease the offset for tick labels on all axes
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5

    # Adjust the offset for tick labels on all axes
    ax.tick_params(axis='x', pad=0.5, length=2)
    ax.tick_params(axis='y', pad=0.5, length=2)

    plt.axhline(y=0, color='silver', linestyle='-', linewidth=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(-0.3, 80, round(ttest_control[1], 3))
    plt.text(1.7, 80, round(ttest_dd[1], 3))
    plt.tight_layout()
    plt.savefig(
        os.path.join(rootdir, "speed_figure_ld-dd_dn_stripplot_10-14h_{0}.pdf".format(species_f.replace(' ', '-'))),
        dpi=350)
    plt.close()
    return


def plot_ld_dd_dn_dif_stripplot(rootdir, spd_aves_condition, custom_palette, custom_order, SMALLEST_SIZE, species_f):
    plt.figure(figsize=(1, 1))
    ax = sns.stripplot(data=spd_aves_condition, x='epoch', y='spd_ave_dif', s=2,
                       palette=custom_palette,
                       order=custom_order)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([-50, 28])
    plt.ylabel("Speed (mm/s)", fontsize=SMALLEST_SIZE)
    plt.title(species_f, fontsize=SMALLEST_SIZE)

    # Decrease the offset for tick labels on all axes
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 0.5

    # Adjust the offset for tick labels on all axes
    ax.tick_params(axis='x', pad=0.5, length=2)
    ax.tick_params(axis='y', pad=0.5, length=2)

    plt.axhline(y=0, color='silver', linestyle='-', linewidth=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(
        os.path.join(rootdir,
                     "speed_figure_ld-dd_dif_stripplot_10-14h_{0}.pdf".format(species_f.replace(' ', '-'))),
        dpi=350)
    plt.close()
    return