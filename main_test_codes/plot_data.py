from main_test_codes.load_data import load_data_dict, load_specific_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot(file_folder, plot_type, plot_params, plot_agent=None, raw_data=False, name_dict=None, n_bs=None):
    data_dict = load_data_dict(file_folder=file_folder)
    if not raw_data:
        data = load_specific_data(data_dict=data_dict, raw_data=raw_data, name_dict=name_dict)
    else:
        data = load_specific_data(data_dict=data_dict, raw_data=raw_data, name_dict=name_dict, n_bs=n_bs)

    if plot_agent == 'seaborn':
        if plot_type == 'hist':
            seaborn_hist(data=data, plot_params=plot_params, plot_title='aiaiaiaiai')
        elif plot_type == 'curve':
            pass
    else:  # in this case, if the plot is not informed, pyplot will be used
        if plot_type == 'hist':
            pass
        elif plot_type == 'curve':
            pass


def multi_plot(files_folder, plot_type, plot_params):
    data_dict = []
    for file_folder in files_folder:
        data_dict.append(load_data_dict())

def seaborn_hist(data, plot_params, plot_title):
    bins = plot_params['bins']
    binrange = plot_params['binrange']
    stat = plot_params['stat']
    kde = plot_params['kde']
    dpi = plot_params['dpi']
    y_min = plot_params['ylim'][0]
    y_max = plot_params['ylim'][1]


    f1 = plt.figure(8, dpi=dpi)
    sns.histplot(data=data, bins=bins, binrange=binrange, stat=stat, kde=kde)
    # sns.histplot(data=data, bins=100, binrange=(0, 140), stat='probability', kde=True)
    plt.title(plot_title)
    plt.ylim(y_min, y_max)
    plt.show()