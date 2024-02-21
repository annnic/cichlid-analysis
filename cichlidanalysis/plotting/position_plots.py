import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from cichlidanalysis.utils.species_names import add_species_from_FishID
from cichlidanalysis.io.meta import check_fish_species


def spd_vs_y(meta, fish_tracks_30m, fish_IDs, rootdir):
    # speed vs Y position, for each fish, for combine fish of species, separated between day and night

    for idx, species in enumerate(meta.loc["species"].unique()):
        # vector for combining all fish of species x
        vp_sp_30m, sf_sp_30m = [], []
        vp_sp_30m_day, sf_sp_30m_day = [], []
        vp_sp_30m_night, sf_sp_30m_night = [], []

        for fish in fish_IDs:
            # 30min binned time resolution
            vertical_pos_30m = fish_tracks_30m[fish_tracks_30m.FishID == fish].vertical_pos

            speed_fish_30m = fish_tracks_30m[fish_tracks_30m.FishID == fish].speed_mm
            full_data_30m = ~np.isnan(vertical_pos_30m) & ~np.isnan(speed_fish_30m)

            vp_30m = vertical_pos_30m[full_data_30m]
            sf_30m = speed_fish_30m[full_data_30m]

            vp_sp_30m = np.concatenate([vp_sp_30m, vp_30m])
            sf_sp_30m = np.concatenate([sf_sp_30m, sf_30m])

            # #### 30min binned time resolution ##### #
            # get vertical position for day and night for fish
            vertical_pos_30m_day = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                   (fish_tracks_30m.daynight == 'd')].vertical_pos

            speed_fish_30m_day = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                 (fish_tracks_30m.daynight == 'd')].speed_mm
            full_data_30m_day = ~np.isnan(vertical_pos_30m_day) & ~np.isnan(speed_fish_30m_day)

            vp_sp_30m_day = np.concatenate([vp_sp_30m_day, vertical_pos_30m_day[full_data_30m_day]])
            sf_sp_30m_day = np.concatenate([sf_sp_30m_day, speed_fish_30m_day[full_data_30m_day]])

            vertical_pos_30m_night = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                     (fish_tracks_30m.daynight == 'n')].vertical_pos

            speed_fish_30m_night = fish_tracks_30m[(fish_tracks_30m.FishID == fish) &
                                                   (fish_tracks_30m.daynight == 'n')].speed_mm
            full_data_30m_night = ~np.isnan(vertical_pos_30m_night) & ~np.isnan(speed_fish_30m_night)

            vp_sp_30m_night = np.concatenate([vp_sp_30m_night, vertical_pos_30m_night[full_data_30m_night]])
            sf_sp_30m_night = np.concatenate([sf_sp_30m_night, speed_fish_30m_night[full_data_30m_night]])

        # for day and night for 30m data
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(sf_sp_30m_day, vp_sp_30m_day, linestyle='', marker='o', markersize=1, color='r', alpha=0.25, label='Day')
        ax2.plot(sf_sp_30m_night, vp_sp_30m_night, linestyle='', marker='o', markersize=1, color='b', alpha=0.25, label='Night')
        ax2.set_xlim([0, 60])
        ax2.set_ylim([0, 1])
        plt.xlabel("Speed mm/s")
        plt.ylabel("Vertical position")
        ax2.legend()
        fig2.suptitle(species)
        fig2.savefig(os.path.join(rootdir, "spd_vs_y_30min_DN.png"))


def plot_position_maps_individuals(meta, fish_tracks, rootdir):
    """ Plots individual heatmaps for day and night for each species"""

    for idx, species in enumerate(meta.loc["species"].unique()):

        fish_IDs = fish_tracks['FishID'].unique()
        # find fish of species
        fish_for_plot = check_fish_species(fish_IDs, species)

        fig2, ax2 = plt.subplots(2, len(fish_for_plot))
        fig2.suptitle("Individual fish averages for {}".format(species))

        for idx1, fish in enumerate(fish_for_plot):
            position_day_x_fish = fish_tracks.loc[(fish_tracks.daynight == 'd') & (fish_tracks.FishID == fish), ["horizontal_pos"]].to_numpy()[:,0]
            position_day_y_fish = fish_tracks.loc[(fish_tracks.daynight == 'd') & (fish_tracks.FishID == fish), ["vertical_pos"]].to_numpy()[:,0]
            position_night_x_fish = fish_tracks.loc[(fish_tracks.daynight == 'n') & (fish_tracks.FishID == fish), ["horizontal_pos"]].to_numpy()[:,0]
            position_night_y_fish = fish_tracks.loc[(fish_tracks.daynight == 'n') & (fish_tracks.FishID == fish), ["vertical_pos"]].to_numpy()[:,0]

            position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day_x_fish, position_day_y_fish,
                                                            bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])
            position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(position_night_x_fish, position_night_y_fish,
                                                            bins=[3, 10], cmap='inferno', range=[[0, 1], [0, 1]])

            # normalise 2dhist
            position_day_xy = position_day_xy/sum(sum(position_day_xy))
            position_night_xy = position_night_xy/sum(sum(position_night_xy))

            ax2[0, idx1].imshow(position_day_xy.T, vmin=0, vmax=0.2)
            ax2[0, idx1].invert_yaxis()
            ax2[0, idx1].get_xaxis().set_ticks([])
            ax2[0, idx1].get_yaxis().set_ticks([])

            ax2[1, idx1].clear()
            ax2[1, idx1].imshow(position_night_xy.T, vmin=0, vmax=0.2)
            ax2[1, idx1].get_xaxis().set_ticks([])
            ax2[1, idx1].get_yaxis().set_ticks([])
            ax2[1, idx1].invert_yaxis()
            if idx1 == 0:
                ax2[0, idx1].set_ylabel("Day")
                ax2[1, idx1].set_ylabel("Night")
        # colorbar?
        plt.savefig(os.path.join(rootdir, "xy_ave_DN_individuals_{0}.png".format(species.replace(' ', '-'))))


def plot_position_maps(meta, fish_tracks, rootdir):
    """ Average plot for day/night for each species
    :param meta:
    :param fish_tracks:
    :param rootdir:
    :return:
    """
    metat = meta.transpose()

    for idx, species_n in enumerate(meta.loc["species"].unique()):

        # find fish_ID of fish of species "species"
        fishes_of_species = metat.loc[metat.species == species_n].index.values

        # split data into day and night
        position_day = fish_tracks.loc[(fish_tracks.daynight == 'd') & (fish_tracks["FishID"].isin(fishes_of_species))]
        position_night = fish_tracks.loc[(fish_tracks.daynight == 'n') & (fish_tracks["FishID"].isin(fishes_of_species))]


        # Creating bins
        x_min = 0
        x_max = np.nanmax(position_day.horizontal_pos)

        y_min = 0
        y_max = np.nanmax(position_day.vertical_pos)

        x_bins = np.linspace(x_min, x_max, 4)
        y_bins = np.linspace(y_min, y_max, 11)

        # creating 2D hist plot
        # day
        fig3 = plt.figure(figsize=(4, 4))
        position_day_xy, xedges_day, yedges_day, _ = plt.hist2d(position_day.horizontal_pos[~np.isnan(
            position_day.horizontal_pos)], position_day.vertical_pos[~np.isnan(position_day.vertical_pos)],
                                                                cmap='inferno', bins=[x_bins, y_bins])
        plt.close(fig3)

        # night
        fig4 = plt.figure(figsize=(4, 4))
        position_night_xy, xedges_night, yedges_night, _ = plt.hist2d(position_night.horizontal_pos[~np.isnan(
            position_night.horizontal_pos)], position_night.vertical_pos[~np.isnan(position_night.vertical_pos)],
                                                                      bins=[3, 10], cmap='inferno')
        plt.close(fig4)

        # properly normalise by counts to get frequency
        position_day_xy = (position_day_xy / sum(sum(position_day_xy)))*100
        position_night_xy = (position_night_xy / sum(sum(position_night_xy))) * 100

        # plot day
        fig3 = plt.figure(figsize=(4, 4))
        plt.imshow(position_day_xy.T, cmap='inferno', vmin=0, vmax=25)
        plt.title("Day")
        cbar = plt.colorbar(label="% occupancy")
        plt.gca().invert_yaxis()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.savefig(os.path.join(rootdir, "xy_ave_Day_{0}.png".format(species_n.replace(' ', '-'))))

        # plot night
        fig4 = plt.figure(figsize=(4, 4))
        plt.imshow(position_night_xy.T, cmap='inferno', vmin=0, vmax=25)
        plt.title("Night")
        cbar = plt.colorbar(label="% occupancy")
        plt.gca().invert_yaxis()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.savefig(os.path.join(rootdir, "xy_ave_Night_{0}.png".format(species_n.replace(' ', '-'))))

        fig1, ax1 = plt.subplots(2, 1)
        # find better way to deal with lack of second dimension when only one species
        if len(meta.loc["species"].unique()) == 1:
            ax1[0].set_title(species_n)
            ax1[0].set_ylabel("Day")
            ax1[0].imshow(position_day_xy.T, cmap='inferno', vmin=0, vmax=25)
            ax1[0].invert_yaxis()
            ax1[0].get_xaxis().set_ticks([])
            ax1[0].get_yaxis().set_ticks([])
            ax1[1].clear()
            ax1[1].imshow(position_night_xy.T, cmap='inferno', vmin=0, vmax=25)
            ax1[1].get_xaxis().set_ticks([])
            ax1[1].get_yaxis().set_ticks([])
            ax1[1].invert_yaxis()
            ax1[1].set_ylabel("Night")
        else:
            ax1[0].set_title(species_n)
            ax1[0].set_ylabel("Day")
            ax1[0].imshow(position_day_xy.T, vmin=0, vmax=25)
            ax1[0].invert_yaxis()
            ax1[0].get_xaxis().set_ticks([])
            ax1[0].get_yaxis().set_ticks([])
            ax1[1].clear()
            ax1[1].imshow(position_night_xy.T, vmin=0, vmax=25)
            ax1[1].get_xaxis().set_ticks([])
            ax1[1].get_yaxis().set_ticks([])
            ax1[1].invert_yaxis()
            ax1[1].set_ylabel("Night")

        fig1.savefig(os.path.join(rootdir, "xy_ave_DN_all_{}.png".format(species_n)))
    return


def plot_combined_v_position(rootdir, fish_tracks_ds, fish_diel_patterns):
    """ Bar plot of day and night vertical position means for each species sorted by the day/night preference

    :param rootdir:
    :param fish_tracks_ds:
    :param fish_diel_patterns:
    :return:
    """
    fish_tracks_dn = fish_tracks_ds.groupby(['daynight', 'FishID', 'species', 'cluster_pattern']).mean().reset_index()

    # dn_index = fish_tracks_dn.groupby(by=['species', 'daynight']).median().reset_index()
    # sorted_index = dn_index.drop(dn_index[dn_index.daynight == 'd'].index).set_index('species').sort_values(by='vertical_pos').index
    sorted_index = fish_diel_patterns.groupby('species').median().sort_values(by='day_night_dif').index

    grped_bplot = sns.catplot(x='species',
                              y='vertical_pos',
                              kind="box",
                              height=6,
                              aspect=4,
                              legend=False,
                              hue='daynight',
                              boxprops=dict(alpha=0.7),
                              fliersize=1,
                              order=sorted_index,
                              data=fish_tracks_dn,
                              palette="bwr_r")

    cmap = plt.cm.get_cmap('bwr_r')
    grped_bplot = sns.stripplot(x='species',
                                y='vertical_pos',
                                hue='daynight',
                                data=fish_tracks_dn,
                                order=sorted_index,
                                palette=[cmap(0), cmap(1000)],
                                size=3,
                                dodge=True)
    grped_bplot.set_xticklabels(labels=sorted_index, rotation=90)
    grped_bplot.set(ylabel='Vertical position', xlabel='Species')
    grped_bplot.set(ylim=[0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, "species_vertical_pos_30min_box_strip_{0}.png".format(dt.date.today())))

    # for subset of the species that fall into a certain cluster pattern
    for key in ['nocturnal', 'diurnal', 'crepuscular', 'undefined']:
        grped_bplot = sns.catplot(x='species',
                                  y='vertical_pos',
                                  kind="box",
                                  height=6,
                                  aspect=4,
                                  legend=False,
                                  hue='daynight',
                                  boxprops=dict(alpha=0.7),
                                  fliersize=0,
                                  data=fish_tracks_dn.loc[fish_tracks_dn.cluster_pattern == key, :],
                                  palette="bwr_r")

        cmap = plt.cm.get_cmap('bwr_r')
        grped_bplot = sns.stripplot(x='species',
                                    y='vertical_pos',
                                    hue='daynight',
                                    data=fish_tracks_dn.loc[fish_tracks_dn.cluster_pattern == key, :],
                                    palette=[cmap(0), cmap(1000)],
                                    size=3,
                                    dodge=True)
        # grped_bplot.set_xticklabels(labels=sorted_index, rotation=90)
        grped_bplot.set(ylabel='Vertical position', xlabel='Species')
        grped_bplot.set(ylim=[0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "species_vertical_pos_30min_box_strip_{0}.png".format(key)))
    return


def plot_v_position_hists(rootdir, vp_hist):
    vp_hist = add_species_from_FishID(vp_hist)

    bins = np.arange(0, 1.1, 0.1)
    bins = np.round(bins, 2).tolist()

    yticks = np.linspace(0.5, 9.5, 10)
    yticklabels = ['Top', '', '', '', '', '', '', '', '', 'Bottom']

    # individial fish histograms
    for state in ['rest', 'non_rest']:
        fig, ax = plt.subplots()
        plot = sns.heatmap(vp_hist.pivot('FishID', 'bin', state).T.iloc[::-1],
                           yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1],
                           vmin=0, vmax=0.6,
                           cmap='RdPu')
        num_fish = np.shape(vp_hist.loc[vp_hist.bin == 0, 'species_six'])[0]
        plot.set_xticks(np.linspace(0, num_fish -1, num_fish))
        plot.set_xticklabels(labels=vp_hist.loc[vp_hist.bin == 0, 'species_six'], rotation=90)
        plot.set_title(state.capitalize())
        plot.set_yticks(yticks)
        plot.set_yticklabels(labels=yticklabels, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "vp_{}_individuals.png".format(state)))

    # averaged histograms
    vp_hist_ave = vp_hist.groupby(['species_six', 'bin']).mean().reset_index()
    for state in ['rest', 'non_rest']:
        fig, ax = plt.subplots()
        plot = sns.heatmap(vp_hist_ave.pivot('bin', 'species_six', state).iloc[::-1],
                           yticklabels=(np.round(bins[0:-1], 2).tolist())[::-1],
                           vmin=0, vmax=0.6,
                           cmap='RdPu',
                           cbar_kws={'label': 'Position frequency'})
        num_species = vp_hist_ave.species_six.nunique()
        plot.set_xticks(np.linspace(0.5, 0.5 + num_species -1, num_species))
        plot.set_xticklabels(labels=vp_hist_ave.pivot('bin', 'species_six', state).columns, rotation=90)
        plot.set_title(state.capitalize())
        plot.set_yticks(yticks)
        plot.set_yticklabels(labels=yticklabels, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(rootdir, "vp_{}_ave.png".format(state)))

    # clustered heatmaps
    for state in ['rest', 'non_rest']:
        individ_corr = vp_hist.pivot('FishID', 'bin', state).T.corr()
        ax = sns.clustermap(individ_corr, figsize=(7, 5), method='single', metric='euclidean', vmin=-1, vmax=1,
                            cmap='RdBu_r', xticklabels=False, yticklabels=True)
        ax.fig.suptitle(state)
