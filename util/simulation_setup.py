import numpy as np
from util.data_management import save_data, load_data, macel_data_dict, write_conf, \
    temp_data_save, temp_data_load, temp_data_delete, convert_file_path_os
from map import Map
from util.param_data_management import load_param, update_param
import tqdm, time
import socket
from util.plot_data_new import plot_histograms, plot_curves, plot_surfaces
from util.mann_whitney_u import compare_dist
from clustering import Cluster

# Just a set of auxiliary functions to setup a simulation environment

def simulate_macel(args):  # todo - fix the and check all the options here
    n_bs = args[0]
    macel = args[1]
    n_samples = args[2]
    n_centers = args[3]
    ue_dist_type = args[4]
    random_centers = args[5]

    if macel.map is not None:  # checking if map data is to be used
        macel.map.generate_samples(n_samples=n_samples)
        macel.grid = macel.map.make_grid()
        macel.del_map()  # deleting the map instance to save some memory
    else:
        macel.grid.make_points(dist_type=ue_dist_type, samples=n_samples, n_centers=n_centers, random_centers=random_centers,
                               plot=False)  # distributing points around centers in the grid
    #print('Parada obrigatoria de teste!')
    macel.set_ue()
    # FALTA TESTAR! CASO SEJA RASTER, grid.point_condition n vai ser none, caso contrário será
    #macel.set_ue(user_condition = macel.grid.point_condition) # FALTA TESTAR! CASO SEJA RASTER grid.point_condition n vai ser none
    
    # snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=n_bs, output_typ='complete', clustering=True)
    output = macel.place_and_configure_bs(n_centers=n_bs)
    # snr_cap_stats = macel.place_and_configure_bs(n_centers=n_bs, output_typ='simple', clustering=False)
    return output


def create_enviroment(parameters, param_path):
    # this function creates the objects and the relationships necessary to run a simulation in simulate_macel_downlink
    from make_grid import Grid
    from antennas.ITU2101_Element import Element_ITU2101
    from antennas.beamforming import Beamforming_Antenna
    from base_station import BaseStation
    from macel import Macel
    from make_raster import Raster
  
    map_ = None  # defining a empty variable to recieve a map class that also can be checked inside the pool
    if parameters['roi_param']['grid']:  # the function checks first if a grid is defined
        print('FOI O GRID')
        grid = Grid()  # grid object
        grid.make_grid(lines=parameters['roi_param']['grid_lines'],
                       columns=parameters['roi_param']['grid_columns'])
        cell_size = parameters['roi_param']['cel_size']

    elif parameters['roi_param']['raster']: #Check if raster type is selected
        grid = Raster(input_shapefile = parameters['roi_param']['input_shapefile'],
        output_raster = parameters['roi_param']['output_raster'],
        projection = parameters['roi_param']['projection'],
        burner_value = parameters['roi_param']['burner_value'])

        print('Rasterizando o shapefile ...')
        grid.rasterize_shapefile()
        grid.make_grid()
        grid.delete_tif_file()
        cell_size = parameters['roi_param']['cel_size']
        print('... feito!')
        
    elif parameters['roi_param']['map']:  # if a grid is not used, it checks if a map is selected
        print('FOI O MAPA')
        folder = 'map_data'
        map_ = Map()
        map_.load(path=convert_file_path_os(folder + '\\30m.pkl'))
        map_.load_general_map_info(path=convert_file_path_os(folder + '\\Brasil_Sce_2010.csv'),
                                   id_column='Cod_Setor', delimiter=';')
        map_.clip_shape(shape=map_.idx_mtx, criteria=parameters['roi_param']['filter_name'],
                       var=parameters['roi_param']['filter_type'], save=True, plot=False)
        # idx_map, mask = map_.clip_shape(shape=map_.idx_mtx, criteria='Tijuca', var='Nm_Bairro',
        #                                 save=True, plot=True)
        # wgt_map = map_.apply_mask(shape=map_.wgt_mtx, mask=mask, plot=True)
        # dst_map = map_.apply_mask(shape=map_.dst_mtx, mask=mask, plot=True)
        map_.generate_samples(n_samples=1000, plot=False)
        # grid = map_.make_grid()
        map_.clear_general_map_info()
        map_.clear_shape_data()
        cell_size = map_.resolution
        grid = None
    else:  # if neither map nor grid is used, raise an exception
        raise NameError('To start a simulation, need to use a GRID, RASTER or MAP in parameter file')

    # instantiating an antenna element
    element = Element_ITU2101(max_gain=parameters['antenna_param']['max_element_gain'],
                              phi_3db=parameters['antenna_param']['phi_3db'],
                              theta_3db=parameters['antenna_param']['theta_3db'],
                              front_back_h=parameters['antenna_param']['front_back_h'],
                              sla_v=parameters['antenna_param']['sla_v'],
                              plot=False)

    # instantiating a beamforming antenna
    beam_ant = Beamforming_Antenna(ant_element=element,
                                   frequency=None,
                                   n_rows=parameters['antenna_param']['n_rows'],
                                   n_columns=parameters['antenna_param']['n_columns'],
                                   horizontal_spacing=parameters['antenna_param']['horizontal_spacing'],
                                   vertical_spacing=parameters['antenna_param']['vertical_spacing'])

    # instantiating a basestation
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

    # checking if downlink or uplink are to be used and picking the parameters
    if parameters['macel_param']['uplink']:
        downlink_specs = parameters['downlink_scheduler']
    else:
        downlink_specs = None
    if parameters['macel_param']['downlink']:
        uplink_specs = parameters['uplink_scheduler']
    else:
        uplink_specs = None

    # instantiating a macrocelular network
    macel = Macel(grid=grid,
                  prop_model=parameters['macel_param']['prop_model'],
                  criteria=parameters['downlink_scheduler']['criteria'],
                  cell_size=cell_size,  # todo - ARRUMAR ISSO AQUI (passar para o grid)!!!
                  base_station=base_station,
                  simulation_time=parameters['macel_param']['time_slots'],
                  time_slot=parameters['macel_param']['time_slot_lngt'],
                  t_min=parameters['downlink_scheduler']['t_min'],
                  scheduler_typ=parameters['downlink_scheduler']['scheduler_typ'],
                  output_type=parameters['exec_param']['output_type'],
                  bw_slot=parameters['downlink_scheduler']['bw_slot'],
                  tdd_up_time=parameters['macel_param']['mux_tdd_up_time'],
                  bs_allocation_typ=parameters['macel_param']['bs_allocation_typ'],
                  downlink_specs=downlink_specs,
                  uplink_specs=uplink_specs)
    macel.set_ue(hrx=parameters['ue_param']['hrx'], tx_power=parameters['ue_param']['tx_power'])
    macel.set_map(map_)
    macel.cluster = Cluster()

    # if a BS file point is used, it sets the centers outside the main simulation to optimize the process inside the pool
    if parameters['macel_param']['bs_allocation_typ'] == 'file':
        n_bs = macel.cluster.from_file(name_file=parameters['macel_param']['bs_location_file_name'])
        parameters['macel_param']['min_bs'] = n_bs
        parameters['macel_param']['max_bs'] = n_bs
        # update_param(param=parameters, param_path=param_path)

    return macel, parameters


def prep_multiproc(threads):
    # this function is just to create a pool and avoid problems with the configuration of it
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
    # this function gets addition information about the simulation and the environment to complement
    # output execution parameters
    path, folder, name_file = save_data()  # storing the path used to save in all iterations
    data_dict = macel_data_dict()  # creating the structure of the output dictionary
    write_conf(folder=folder, parameters=global_parameters, yml_file=param_path)  # saving the param/stats files

    global_parameters['exec_param']['simulation_time'] = []
    global_parameters['exec_param']['executed_n_bs'] = []
    global_parameters['exec_param']['executed_n_ue'] = []
    global_parameters['exec_param']['threads'] = process_pool._processes
    global_parameters['exec_param']['PC_ID'] = socket.gethostname()

    return global_parameters, path, folder, name_file, data_dict

def update_sim_param(parameter, n_cells, n_samples, initial_time):
    # updating the parameters to be write on the exec_stats file
    parameter['exec_param']['executed_n_bs'].append(n_cells)
    parameter['exec_param']['simulation_time'].append(
        np.round((time.time() - initial_time) / 60, decimals=2))  # simulation time (in minutes)
    if parameter['macel_param']['ue_dist_typ'] != 'uniform':  # to check if n_samples is used
        parameter['exec_param']['executed_n_ue'].append(n_samples * parameter['macel_param']['n_centers'])
    else:
        parameter['exec_param']['executed_n_ue'].append(n_samples)

    return parameter


def check_iter_type(iter_params):
    bs_range_chk = not(not iter_params['bs_step'] or iter_params['bs_step'] == 0)
    ue_range_chk = not(not iter_params['samples_step'] or
                   iter_params['samples_step'] == 0)
    if bs_range_chk and ue_range_chk:
        raise ValueError('The simulation can only iterate for different sample or BS values! Please check the bs_step and samples_step on the parameter file !!!')
    elif not bs_range_chk and not ue_range_chk:  # treat as if iterating over a BS range
        n_bs = iter_params['min_bs']
        n_samples = None
        iter_range = range(iter_params['min_bs'], iter_params['min_bs'])
        iter_type = None
    elif not bs_range_chk:
        n_bs = iter_params['min_bs']
        n_samples = None
        if iter_params['samples_min'] and iter_params['samples_max']:
            iter_range = range(iter_params['samples_min'],
                               iter_params['samples_max'] + 1,
                               iter_params['samples_step'])
            iter_type = 'UE'
        else:
            raise ValueError('Need to set all range of values to iterate over UE samples: samples_min and samples_max')

    else:
        n_samples = iter_params['samples_min']
        n_bs = None
        if iter_params['min_bs'] and iter_params['max_bs']:
            iter_range = range(iter_params['min_bs'],
                               iter_params['max_bs'] + 1,
                               iter_params['bs_step'])
            iter_type = 'BS'
        else:
            raise ValueError('Need to set all range of values to iterate over BS numbers: min_bs and max_bs')

    return iter_range, iter_type, n_bs, n_samples


def start_simmulation(conf_file):
    global_parameters, param_path = load_param(filename=conf_file, backup=True)

    process_pool = prep_multiproc(threads=global_parameters['exec_param']['threads'])
    global_parameters, path, folder, name_file, data_dict = get_additional_sim_param(global_parameters=global_parameters,
                                                               param_path=param_path, process_pool=process_pool)

    temp_data_save(zero_state=True)  # creating or cleaning the temp folder

    # separating parameters to pass the minimum data to the pool
    # n_samples = global_parameters['macel_param']['n_samples']
    n_centers = global_parameters['macel_param']['n_centers']
    max_iter = global_parameters['exec_param']['max_iter']
    batch_size = global_parameters['exec_param']['batch_size']
    ue_dist_typ = global_parameters['macel_param']['ue_dist_typ']
    center_dist_typ = global_parameters['macel_param']['center_distribution']
    hypothesis_test = global_parameters['exec_param']['hypothesis_test']
    hypothesis_test_var = global_parameters['exec_param']['hypothesis_test_var']

    if center_dist_typ == 'uniform':
        random_centers = True
    elif center_dist_typ == 'cluster':
        random_centers = False
    else:
        raise TypeError("not valid center_distribution option")

    bs_vec = []
    macel, global_parameters = create_enviroment(parameters=global_parameters, param_path=folder)

    div_floor = max_iter//batch_size
    div_dec = max_iter % batch_size
    if div_dec != 0:
        steps = np.zeros(shape=div_floor + 1, dtype='int') + batch_size
        steps[-1] = div_dec
    else:
        steps = np.zeros(shape=div_floor, dtype='int') + batch_size

    iter_range, iter_type, n_cells, n_samples = check_iter_type(global_parameters['macel_param'])

    for iter_var in iter_range:
        if iter_type == 'BS':
            n_cells = iter_var
        elif iter_type == 'UE':
            n_samples = iter_var


    # for n_cells in range(global_parameters['macel_param']['min_bs'], global_parameters['macel_param']['max_bs'] + 1):
        print('\nrunning with ', n_cells, ' BSs, ', n_samples, 'UEs and a batch size of', batch_size, 'iterations')

        initial_time = time.time()  # this is used to write the simulation time on the exec_stats .txt file

        bs_vec.append(n_cells)

        i = 0
        # data = []
        # for sub_iter in steps:
        end_sim = False
        iter = 0
        while end_sim is not True:
            # starting and finishing the process pool in each iteration to save memory for the other functions
            process_pool = prep_multiproc(threads=global_parameters['exec_param']['threads'])
            i += 1
            iter += batch_size

            print(' ')
            print('Running step ', i, ':')

            data_ = list(
                tqdm.tqdm(
                    process_pool.imap_unordered(
                        simulate_macel, [(n_cells, macel, n_samples, n_centers, ue_dist_typ, random_centers)
                                                  for i in range(batch_size)]), total=round(batch_size)
                ))

            process_pool.terminate()  # to avoid memory overflow when processing the plots

            data = temp_data_load()
            if data and hypothesis_test:
                end_sim = compare_dist(data, data + data_, hypothesis_test_var)

            data = data + data_
            temp_data_save(batch_file={'data': data, 'index': i})  # this will store temporary files on disk and avoid memory consumption

            del data
            del data_
            if iter >= max_iter:
                print('Achieved the max number of iterations')
                break

        data = temp_data_load()
        data_dummy = load_data(name_file=name_file)
        if data_dummy:
            data_dict = data_dummy
            del data_dummy

        data_dict = macel_data_dict(data_dict_=data_dict, data_=data, n_cells=n_cells,
                                    n_samples=n_samples, n_centers=n_centers,
                                    dist_typ=global_parameters['macel_param']['ue_dist_typ'])

        temp_data_delete(type='batch')  # deleting the temporary files because the output dictionary was already created

        save_data(path=path, data_dict=data_dict)  # saving/updating data
        del data_dict
        del data

        # updating the parameters to be write on the exec_stats file
        global_parameters = update_sim_param(parameter=global_parameters, n_cells=n_cells, n_samples=n_samples, initial_time=initial_time)

        write_conf(folder=folder, parameters=global_parameters)

        # plots
        if global_parameters['exec_param']['plot_curves']:
            print('saving curves ....')
            plot_curves(name_file=name_file, max_iter=max_iter,
                        # iter_list=global_parameters['exec_param']['executed_n_bs'],
                        global_parameters=global_parameters, list_typ=iter_type)
            print('saving curves .... [done]')

        if global_parameters['exec_param']['plot_hist']:
            print('saving histograms ....')
            plot_histograms(name_file=name_file, max_iter=max_iter,
                            # iter_list=global_parameters['exec_param']['executed_n_bs'],
                            global_parameters=global_parameters, list_typ=iter_type)
            print('saving histograms .... [done]')

        if global_parameters['exec_param']['plot_surf']:
            print('saving surface plots ....')
            plot_surfaces(name_file=name_file, global_parameters=global_parameters, list_typ=iter_type)
            print('saving surface plots .... [done]')
