import numpy as np
from util.data_management import save_data, load_data, macel_data_dict, write_conf
# from util.plot_data import plot_hist, plot_surface, plot_curves
from util.load_parameters import load_param
import multiprocessing, os, tqdm, time
import socket
from util.plot_data_new import plot_histograms, plot_curves, plot_surfaces

# Just a set of auxiliary functions to setup a simulation environment

def simulate_macel_downlink(args):  # todo - fix the and check all the options here
    n_bs = args[0]
    macel = args[1]
    n_samples = args[2]
    n_centers = args[3]

    macel.grid.make_points(dist_type='gaussian', samples=n_samples, n_centers=n_centers, random_centers=False,
                           plot=False)  # distributing points around centers in the grid
    macel.set_ue()
    # snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=n_bs, output_typ='complete', clustering=True)
    output = macel.place_and_configure_bs(n_centers=n_bs, clustering=True)
    # snr_cap_stats = macel.place_and_configure_bs(n_centers=n_bs, output_typ='simple', clustering=False)
    return output


def create_enviroment(parameters):
    # this function creates the objects and the relationships necessary to run a simulation in simulate_macel_downlink
    from make_grid import Grid
    from antennas.ITU2101_Element import Element_ITU2101
    from antennas.beamforming import Beamforming_Antenna
    from base_station import BaseStation
    from macel import Macel


    grid = Grid()  # grid object
    grid.make_grid(lines=parameters['roi_param']['grid_lines'],
                   columns=parameters['roi_param']['grid_columns'])

    element = Element_ITU2101(max_gain=parameters['antenna_param']['max_element_gain'],
                              phi_3db=parameters['antenna_param']['phi_3db'],
                              theta_3db=parameters['antenna_param']['theta_3db'],
                              front_back_h=parameters['antenna_param']['front_back_h'],
                              sla_v=parameters['antenna_param']['sla_v'],
                              plot=False)

    beam_ant = Beamforming_Antenna(ant_element=element,
                                   frequency=None,
                                   n_rows=parameters['antenna_param']['n_rows'],
                                   n_columns=parameters['antenna_param']['n_columns'],
                                   horizontal_spacing=parameters['antenna_param']['horizontal_spacing'],
                                   vertical_spacing=parameters['antenna_param']['vertical_spacing'])

    base_station = BaseStation(frequency=parameters['bs_param']['freq'],
                               tx_power=parameters['bs_param']['tx_power'],
                               tx_height=parameters['bs_param']['htx'],
                               bw=parameters['bs_param']['bw'],
                               n_sectors=parameters['bs_param']['n_sectors'],
                               antenna=beam_ant,
                               gain=None,
                               downtilts=parameters['bs_param']['downtilt'],
                               plot=False)

    base_station.sector_beam_pointing_configuration(n_beams=parameters['bs_param']['n_beams'])

    if parameters['macel_param']['uplink']:
        downlink_specs = parameters['downlink_scheduler']
    else:
        downlink_specs = None
    if parameters['macel_param']['downlink']:
        uplink_specs = parameters['uplink_scheduler']
    else:
        uplink_specs = None

    macel = Macel(grid=grid,
                  prop_model='free space',
                  criteria=parameters['downlink_scheduler']['criteria'],
                  cell_size=parameters['roi_param']['cel_size'],  # todo - ARRUMAR ISSO AQUI (passar para o grid)!!!
                  base_station=base_station,
                  simulation_time=parameters['macel_param']['time_slots'],
                  time_slot=parameters['macel_param']['time_slot_lngt'],
                  t_min=parameters['downlink_scheduler']['t_min'],
                  scheduler_typ=parameters['downlink_scheduler']['scheduler_typ'],
                  output_type=parameters['exec_param']['output_type'],
                  bw_slot=parameters['downlink_scheduler']['bw_slot'],
                  tdd_up_time=parameters['macel_param']['mux_tdd_up_time'],
                  downlink_specs=downlink_specs,
                  uplink_specs=uplink_specs)
    macel.set_ue(hrx=parameters['ue_param']['hrx'], tx_power=parameters['ue_param']['tx_power'])

    return macel


def prep_multiproc(threads):
    import multiprocessing
    import os

    max_threads = os.cpu_count()

    if threads == 0:
        threads = max_threads - 1
    elif threads > max_threads:
        threads = max_threads - 1
        print('The selected number of threads is higher than the maximum of the CPU.')
    if threads > 61:  # to run in processors with 30+ cores
        threads = 61
    print('Running with ' + str(threads) + ' threads')
    p = multiprocessing.Pool(processes=threads)

    return p


def get_additional_sim_param(global_parameters, param_path, process_pool):
    path, folder, name_file = save_data()  # storing the path used to save in all iterations
    data_dict = macel_data_dict()  # creating the structure of the output dictonary
    write_conf(folder=folder, parameters=global_parameters, yml_file=param_path)  # saving the param/stats files

    global_parameters['exec_param']['simulation_time'] = []
    global_parameters['exec_param']['executed_n_bs'] = []
    global_parameters['exec_param']['threads'] = process_pool._processes
    global_parameters['exec_param']['PC_ID'] = socket.gethostname()

    return global_parameters, path, folder, name_file, data_dict


def start_simmulation(conf_file):
    global_parameters, param_path = load_param(filename=conf_file, backup=True)

    process_pool = prep_multiproc(threads=global_parameters['exec_param']['threads'])

    global_parameters, path, folder, name_file, data_dict = get_additional_sim_param(global_parameters=global_parameters,
                                                               param_path=param_path, process_pool=process_pool)

    # separating parameters to pass the minimum data to the pool
    n_samples = global_parameters['macel_param']['n_samples']
    n_centers = global_parameters['macel_param']['n_centers']
    max_iter = global_parameters['exec_param']['max_iter']

    bs_vec = []
    macel = create_enviroment(parameters=global_parameters)

    div_floor = max_iter//100
    div_dec = max_iter%100
    if div_dec != 0:
        steps = np.zeros(shape=div_floor + 1, dtype='int') + 100
        steps[-1] = div_dec
    else:
        steps = np.zeros(shape=div_floor, dtype='int') + 100
    last_step = len(steps)

    for n_cells in range(global_parameters['macel_param']['min_bs'], global_parameters['macel_param']['max_bs'] + 1):
        print('running with ', n_cells, ' BSs')

        initial_time = time.time()

        bs_vec.append(n_cells)

        i = 0
        data = []
        for sub_iter in steps:
            i += 1

            print(' ')
            print('Running step ', i, ' of ', last_step, ':')

            data_ = list(
                tqdm.tqdm(
                    process_pool.imap_unordered(simulate_macel_downlink,
                                                   [(n_cells, macel, n_samples, n_centers) for i in
                                                    range(sub_iter)]),
                    total=round(sub_iter)
                ))

            data = data + data_
            data_ = None

        data_dict = macel_data_dict(data_dict_=data_dict, data_=data, n_cells=n_cells)

        save_data(path=path, data_dict=data_dict)  # saving/updating data

        global_parameters['exec_param']['executed_n_bs'].append(n_cells)
        global_parameters['exec_param']['simulation_time'].append(
            np.round((time.time() - initial_time) / 60, decimals=2))  # simulation time (in minutes)

        write_conf(folder=folder, parameters=global_parameters)

        if global_parameters['exec_param']['plot_curves']:
            print('saving curves ....')
            plot_curves(name_file=name_file, max_iter=max_iter, bs_list=global_parameters['exec_param']['executed_n_bs'],
                        global_parameters=global_parameters)
            print('saving curves .... [done]')

        if global_parameters['exec_param']['plot_hist']:
            print('saving histograms ....')
            plot_histograms(name_file=name_file, max_iter=max_iter, global_parameters=global_parameters)  # testing !!!
            print('saving histograms .... [done]')

        if global_parameters['exec_param']['plot_surf']:
            print('saving surface plots ....')
            plot_surfaces(name_file=name_file, global_parameters=global_parameters)
            print('saving surface plots .... [done]')

        data = None
