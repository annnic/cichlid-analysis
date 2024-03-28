import os
import copy
import glob

import pandas as pd

from cichlidanalysis.analysis.processing import add_daytime, ave_daily_fish


def create_fv1(all_species, fish_IDs, fish_tracks, metat, rootdir):
    """feature vector: for each fish readout vector of feature values
    version 1: Day/Night for: speed -  mean, stdev, median; y position - mean, stdev, median;
    FM - mean, stdev, median;
    to add later:
    movement immobile/mobile bout lengths, distributions, max speed
    BS - mean, stdev, median
    BS rest/active bout lengths, distributions, max speed
    30min bins of each data"""
    column_names = ['spd_mean_d', 'spd_mean_n', 'spd_std_d', 'spd_std_n', 'spd_median_d', 'spd_median_n', 'move_mean_d',
                    'move_mean_n', 'move_std_d', 'move_std_n', 'move_median_d', 'move_median_n', 'y_mean_d', 'y_mean_n',
                    'y_std_d', 'y_std_n', 'y_median_d', 'y_median_n', 'fish_length_mm']

    for species in all_species:
        df = pd.DataFrame([],  columns=column_names)

        for fish in fish_IDs:
            fish_v_d = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daynight == "d"),
                                                ["speed_mm", "movement", "vertical_pos"]]
            fish_v_n = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daynight == "n"),
                                                ["speed_mm", "movement", "vertical_pos"]]

            df_f = pd.DataFrame([[fish_v_d.mean()[0], fish_v_n.mean()[0], fish_v_d.std()[0], fish_v_n.std()[0],
                                  fish_v_d.median()[0], fish_v_n.median()[0], fish_v_d.mean()[1], fish_v_n.mean()[1],
                                  fish_v_d.std()[1], fish_v_n.std()[1], fish_v_d.median()[1], fish_v_n.median()[1],
                                  fish_v_d.mean()[2], fish_v_n.mean()[2], fish_v_d.std()[2], fish_v_n.std()[2],
                                  fish_v_d.median()[2], fish_v_n.median()[2], metat.loc[fish, 'fish_length_mm']]],
                                index=[fish], columns=column_names)
            df_f = df_f.round(4)
            df = pd.concat([df, df_f])

        df.to_csv(os.path.join(rootdir, "{}_als_fv.csv".format(species)))
    print("Finished fv v1")


def create_fv2(all_species, fish_tracks, fish_bouts_move, fish_bouts_rest, fish_IDs, metat, fish_tracks_30m, rootdir):
    """ feature vector version  2: for each fish readout vector of feature values
    version 2: 'predawn', 'dawn', 'day', 'dusk', 'postdusk', 'night' for: speed -  mean, stdev; y position - mean, stdev;
    FM - mean, stdev, immobile/mobile bout lengths; Rest - mean, stdev, bout lengths;
    """
    time_v2_m_names = ['predawn', 'dawn', 'day', 'dusk', 'postdusk', 'night']
    times_v2_m = {'predawn': 360, 'dawn': 420, 'day': 480, 'dusk': 1080, 'postdusk': 1140, 'night': 1200}

    fish_tracks = add_daytime(fish_tracks, time_v2_m_names, times_v2_m)
    fish_bouts_move = add_daytime(fish_bouts_move, time_v2_m_names, times_v2_m)
    fish_bouts_rest = add_daytime(fish_bouts_rest, time_v2_m_names, times_v2_m)
    print("added daytime {} column".format(time_v2_m_names))

    data_names = ['spd_mean', 'move_mean', 'rest_mean', 'y_mean', 'spd_std', 'move_std', 'rest_std', 'y_std',
                  'move_bout_mean', 'nonmove_bout_mean', 'rest_bout_mean', 'nonrest_bout_mean', 'move_bout_std',
                  'nonmove_bout_std', 'rest_bout_std', 'nonrest_bout_std']

    for species in all_species:
        new_df = True
        for fish in fish_IDs:
            new_fish = True
            for epoque in time_v2_m_names:
                # adding main data
                column_names = [sub + ('_' + epoque) for sub in data_names]
                fish_v = fish_tracks.loc[(fish_tracks.FishID == fish) & (fish_tracks.daytime == epoque), ['speed_mm',
                                                                                    'movement', 'rest', 'vertical_pos']]
                fish_b_move = fish_bouts_move.loc[(fish_bouts_move.FishID == fish) & (fish_bouts_move.daytime == epoque),
                                                  ['movement_len', 'nonmovement_len']]
                fish_b_rest = fish_bouts_rest.loc[(fish_bouts_rest.FishID == fish) & (fish_bouts_rest.daytime == epoque),
                                                  ['rest_len', 'nonrest_len']]

                # make dataframe for this epoque
                df_e = pd.DataFrame([[fish_v.mean()['speed_mm'], fish_v.mean()['movement'], fish_v.mean()['rest'],
                                      fish_v.mean()['vertical_pos'], fish_v.std()['speed_mm'], fish_v.std()['movement'],
                                      fish_v.std()['rest'], fish_v.std()['vertical_pos'],
                                      fish_b_move.mean()['movement_len'].total_seconds(),
                                      fish_b_move.mean()['nonmovement_len'].total_seconds(),
                                      fish_b_rest.mean()['rest_len'].total_seconds(),
                                      fish_b_rest.mean()['nonrest_len'].total_seconds(),
                                      fish_b_move.std()['movement_len'].total_seconds(),
                                      fish_b_move.std()['nonmovement_len'].total_seconds(),
                                      fish_b_rest.std()['rest_len'].total_seconds(),
                                      fish_b_rest.std()['nonrest_len'].total_seconds()]], index=[fish],
                                      columns=column_names)
                # add epoques together
                if new_fish:
                    df_f = copy.copy(df_e)
                    new_fish = False
                else:
                    df_f = pd.concat([df_f, df_e], axis=1)
            df_f['fish_length_mm'] = metat.loc[fish, 'fish_length_mm']
            _, _, df_f['total_rest'] = ave_daily_fish(fish_tracks_30m, fish, 'rest')
            df_f = df_f.round(2)

            # add fish together
            if new_df:
                feature_vector = copy.copy(df_f)
                new_df = False
            else:
                feature_vector = pd.concat([feature_vector, df_f], axis=0)

        feature_vector.to_csv(os.path.join(rootdir, "{}_als_fv2.csv".format(species)))
    print("Saved out feature vector v2")
    print("Finished")


def load_feature_vectors(folder, suffix="*als_fv.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()
    first_done = 0

    for file in files:
        if first_done:
            data_s = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            data = pd.concat([data, data_s])

        else:
            # initiate data frames for each of the fish, beside the time series,
            data = pd.read_csv(os.path.join(folder, file), sep=',')
            print("loaded file {}".format(file))
            first_done = 1

    data = data.rename(columns={"Unnamed: 0": "fish_ID"})
    data = data.reset_index(drop=True)

    print("All down sampled als.csv files loaded")
    return data


def load_diel_pattern(folder, suffix="*dp.csv"):
    os.chdir(folder)
    files = glob.glob(suffix)
    files.sort()

    # use most up to date file
    data = pd.read_csv(os.path.join(folder, files[-1]), sep=',')

    print("Most up to date file loaded")
    return data
