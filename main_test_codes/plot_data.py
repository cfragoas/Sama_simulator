from load_data import load_data_dict, load_specific_data

import matplotlib.pyplot as plt
import seaborn as sns

def plot(file_folder, plot_type, plot_params, plot_agent=None, raw_data=False):
    data_dict = load_data_dict(file_folder=file_folder)
    if raw_data:
        data = load_specific_data(data_dict=data_dict, raw_data=raw_data, name_dict=name_dict)
    else:
        data = load_specific_data(data_dict=data_dict, raw_data=raw_data, name_dict=name_dict, n_bs=n_bs)

    if plot_agent is 'seaborn':
        if plot_type == 'hist':
            pass
        elif plot_type == 'curve':
            pass
    else:  # in this case, if the plot is not informed, pyplot will be used
        pass


def multi_plot(files_folder, plot_type, plot_params):
    data_dict = []
    for file_folder in files_folder:
        data_dict.append(load_data_dict())