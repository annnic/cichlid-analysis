import os

from cichlidanalysis.io.get_file_folder_paths import select_dir_path
from cichlidanalysis.analysis.run_combine_als_simple import combine_binning


if __name__ == '__main__':

    rootdir = select_dir_path()

    # list_of_folders = ['Neobue', 'Neonig', 'Xenbat', 'Varmoo', 'Tylpol']
    # list_of_folders = ['Trobem', 'Trioto', 'Tromoo', 'Telshe', 'Telvit']
    # list_of_folders = ['Telluf', 'Petpol', 'Parnig', 'Ophboo', 'Neoven']
    # list_of_folders = ['Neotoa', 'Neosav', 'Neotre']
    # list_of_folders = ['Neopul-daffodil', 'Neooli']
    # list_of_folders = ['Neomul', 'Neogra', 'Neolon', 'Auldew']
    # list_of_folders = ['Neomar', 'Neokom', 'Neohel', 'Neodev', 'Loblab', 'Lepelo', 'Lepatt', 'Gnapfe', 'Eracya', 'Auldew']
    # list_of_folders = ['NeofaM', 'Neofal', 'Neocyl', 'Neocyg', 'Lamoce', 'Julreg', 'Julorn', 'Juldic', 'Julmar', 'Altshe']
    # list_of_folders = ['Neobri', 'Neocra', 'Neocau', 'Neobre', 'Cypfro', 'Cypcol', 'Calple', 'Ctehor', 'Chacya', 'Altcom']
    list_of_folders = ['Erecya']



    for folder in list_of_folders:
        combine_binning(os.path.join(rootdir, folder), binning_m=30)
