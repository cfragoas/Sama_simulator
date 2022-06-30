import socket

from util.save_data import save_data, load_data, macel_data_dict, write_conf
from util.plot_data import plot_hist, plot_surface, plot_curves
from util.simulation_setup import simulate_macel_downlink, create_enviroment, prep_multiproc
from util.load_parameters import load_param

import multiprocessing, os, tqdm, time
import numpy as np


if __name__ == '__main__':

    conf_file = 'param.yml'

    bs_vec = []

    path, folder = save_data()  # storing the path used to save in all iterations

    global_parameters, param_path = load_param(filename=conf_file, backup=True)

    data_dict = macel_data_dict()  # creating the structure of the output dictonary

    write_conf(folder=folder, parameters=global_parameters, yml_file=param_path)  # saving the param/stats files

    processing_pool = prep_multiproc(threads=global_parameters['exec_param']['threads'])

    global_parameters['exec_param']['simulation_time'] = []
    global_parameters['exec_param']['threads'] = processing_pool._processes
    global_parameters['exec_param']['PC_ID'] = socket.gethostname()

    macel = create_enviroment(parameters=global_parameters)

    # separate parameters to pass the minimum data to the pool
    n_samples = global_parameters['macel_param']['n_samples']
    n_centers = global_parameters['macel_param']['n_centers']
    max_iter = global_parameters['exec_param']['max_iter']

    bs_range = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    # bs_range = [2, 4, 6, 8, 10, 12, 14, 16]

    for n_cells in bs_range:
        print('running with ', n_cells, ' BSs')

        initial_time = time.time()

        # macel.grid.clear_grid()  # added to avoid increasing UE number without intention
        bs_vec.append(n_cells)

        i = 0
        data = []
        for sub_iter in range(0, max_iter, round(max_iter / 20)):
            i += 1

            print(' ')
            print('Running step ', i, ' of ', round(max_iter/(max_iter / 20)), ':')

            data_ = list(
                        tqdm.tqdm(
                            processing_pool.imap_unordered(simulate_macel_downlink,
                                             [(n_cells, macel, n_samples, n_centers) for i in range(round(max_iter / 20))]),
                            total=round(max_iter / 20)
                    ))

            data = data + data_
            data_ = None

        # processing_pool.terminate()
        # processing_pool = prep_multiproc(threads=global_parameters['exec_param']['threads'])

        snr_cap_stats = [x[0] for x in data]
        raw_data = [x[1] for x in data]

        plot_hist(raw_data=raw_data, path=folder, n_bs=n_cells,
                  max_iter=max_iter,
                  criteria=global_parameters['macel_param']['criteria'])

        plot_surface(grid=macel.grid.grid, position=np.concatenate([x['bs_position'] for x in raw_data]),
                     parameter=np.array(snr_cap_stats)[:, 2], path=folder, n_bs=n_cells,
                     max_iter=max_iter)

        data_dict = macel_data_dict(data_dict_=data_dict, data_=data)

        save_data(path=path, data_dict=data_dict)  # saving/updating data

        plot_curves(mean_snr=data_dict['mean_snr'], std_snr=data_dict['std_snr'], mean_cap=data_dict['mean_cap'],
             std_cap=data_dict['std_cap'],
             mean_user_time=data_dict['mean_user_time'], std_user_time=data_dict['std_user_time'],
             mean_user_bw=data_dict['mean_user_bw'],
             std_user_bw=data_dict['std_user_bw'],total_meet_criteria=data_dict['total_meet_criteria'], max_iter=max_iter,
             n_bs_vec=bs_vec, individual=False, path=folder, criteria=global_parameters['macel_param']['criteria'])

        global_parameters['macel_param']['last_n_bs'] = n_cells
        global_parameters['exec_param']['simulation_time'].append(np.round((time.time()-initial_time)/60, decimals=2))  # simulation time (in minutes)

        write_conf(folder=folder, parameters=global_parameters)

        data = None
