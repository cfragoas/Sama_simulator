import numpy as np
from main_test_codes.plot_data import plot, multi_plot

def initiate_plot(plot_type, plot_agent, file_folder):
    # tyoe_plot: hist, curve or surface
    # plot_agent: seaborn or pyplot
    x = plot_type + ' ' + plot_agent

    if x ==  'seaborn hist':
        # seaborn histogram plot
        plot_params = {
            'title': 'título',
            'name_data': 'cap',
            'n_bs': 6,
            'bins': 100,  # number of bins of the histogram
            'binrange': (0, 100),  # não sei !!!!
            'stat': 'probability',  # ver aqui
            'kde': True,  # to plot a probability desity function using KDE (Kernel Density Estimation)
            'dpi': 150,  # dpi of the image
            'ylim': [0, 1]}  # limit for the y axis

        plot(file_folder=file_folder, plot_type=plot_type, plot_params=plot_params, plot_agent=plot_agent,
             raw_data=True, name_dict=plot_params['name_data'], n_bs=plot_params['n_bs'])

    elif x == 'pyplot hist':
        # matplotlib histogram plot
        plot_params = {
            'title': 'título',
            'name_data': 'snr',
            'n_bs': 5,
        }

        plot(file_folder=file_folder, plot_type=plot_type, plot_params=plot_params, plot_agent=plot_agent,
             raw_data=True, name_dict=plot_params['name_data'], n_bs=plot_params['n_bs'])

    elif x ==  'pyplot curve':
        # matplotlib curve plot
        plot_params = {
            'title': 'título',
            'name_data': 'snr',
            'xlabel': 'teste',
            'ylabel': 'teste2',
            'yticks': np.arange(0, 100, 5),
            'color': 'green'
        }

        plot(file_folder=file_folder, plot_type=plot_type, plot_params=plot_params, plot_agent=plot_agent,
             raw_data=False, name_dict=plot_params['name_data'], n_bs=None)

    elif x == 'pyplot surface':
        # matplotlib surface plot
        plot_params = {

        }

        plot(file_folder=file_folder, plot_type=plot_type, plot_params=plot_params, plot_agent=plot_agent,
             raw_data=False, name_dict=plot_params['name_data'], n_bs=None)


