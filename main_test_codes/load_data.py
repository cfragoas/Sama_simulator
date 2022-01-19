import os, platform
import pickle
import glob
import numpy as np

def load_data_dict(file_folder=None):
    # need to change the file name and folder here
    # name_file = '01_02_22-03_57_23'

    folder = os.path.dirname(__file__)
    folder = folder.replace('/', '\\')
    folder = '\\'.join(folder.split('\\')[:-1])
    folder += '\\output\\'
    folder = folder + file_folder + '\\'

    # x =+ name_file + '.pkl'

    if platform.system() == 'Darwin':  # to check if is running on MacOS
        # path = path.replace('\\', '/')
        folder = folder.replace('\\', '/')

    os.chdir(folder)
    name_file = glob.glob('./*.pkl')[0].replace('./', '')

    folder = folder + '\\' + name_file

    print('File and folder: ', folder)

    if platform.system() == 'Darwin':  # to check if is running on MacOS
        # path = path.replace('\\', '/')
        folder = folder.replace('\\', '/')

    with open(folder, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()

    print('load data from ' + name_file)

    return data_dict[0]

def load_specific_data(name_dict=None, raw_data = False, data_dict=None, n_bs=None):
    if data_dict is not None:
        if not raw_data:  # checks if the data is simplified or the raw data
            if n_bs is not None:
                return data_dict[name_dict][n_bs + 1]  # returns the simplified data fro a specfic BSs number
            else:
                return data_dict[name_dict]  # just returns all simplified data (1 to n_bs)

        else:
            if n_bs is not None:
                raw_data = data_dict['raw_data'][n_bs-1]
                data = np.concatenate([x[name_dict] for x in raw_data])
                return data  # returns, from a specfic number of BSs, a specific parementer's raw data
            else:
                return data_dict['raw_data']  # in this case, just returns all raw_data
    else:
        print('Need to pass the data_dict to run!!!')