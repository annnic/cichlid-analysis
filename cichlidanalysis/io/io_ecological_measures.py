import os

import pandas as pd
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import Tk


def get_meta_paths():
    """ Hard coded paths for Ronco data on Annika's macbook

    :return:
    """
    file_path = os.path.realpath(__file__)
    code_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    # paths for Ronco 2021 data
    ronco_data_path = os.path.join(code_path, "example_data", "Fig2_data.csv")
    # https://www.nature.com/articles/s41586-020-2930-4#Sec30

    # path for cichlid meta
    cichlid_meta_path = os.path.join(code_path, "example_data", "cichlid_meta_20231129.csv")

    # voucher_key_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/01_specimen_voucher_key.csv"
    # pigmentation_pattern_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_scores_pigmentation_pattern.csv"
    # stable_isotope_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_stable_isotope_data.csv"
    #
    # lower_pharyngeal_jaw_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_landmark_data_lower_pharyngeal_jaw.tps"
    # body_shape_path = "/Volumes/BZ/RG Schier/Scientific Data/Cichlid-genomes/06_landmark_data_body_shape.tps"

    return ronco_data_path, cichlid_meta_path


if __name__ == '__main__':
    ronco_data_path = get_meta_paths()

    root = Tk()
    root.withdraw()
    root.update()
    vid_dir = askdirectory()
    root.destroy()

    ronco_data = pd.read_csv(ronco_data_path)