import os, datetime, platform, logging, copy
import numpy as np
import pickle
from itertools import product
import pandas as pd
def convert_file_path_os(path):
    # this function simply converts the file path to correspond to the machine OS system
    import platform
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        path = path.replace('\\', '/')
    return path

def write_conf(folder, parameters, yml_file=None):
    # this function saves the execution data into a .txt file on the simulation folder
    with open(folder + 'exec_stats.txt', 'w') as f:
        for key, dict in parameters.items():
            f.writelines(str(key) + '\n \n')
            for key2, item in parameters[key].items():
                f.writelines(str(key2) + ': ' + str(item) + '\n')
            f.writelines('\n')

    if yml_file is not None:  # to save the yaml file (to rerun or resume an execution)
        import shutil
        shutil.copy(yml_file, folder)


def macel_data_dict(data_dict_=None, data_=None, n_cells=None, n_samples=None, n_centers=None, dist_typ=None):
    # this functions creates a dictionary for the simulations results be stored
    # the raw data is not affected by this organization
    if not data_ or not data_dict_:
        # creating the base simplified data dict
        data_dict_ = {'BSs': [], 'UEs': [], 'mean_snr': [], 'std_snr': [], 'mean_cap': [], 'std_cap': [], 'mean_user_time': [],
                      'std_user_time': [], 'mean_user_bw': [], 'std_user_bw': [], 'raw_data': [], 'total_meet_criteria': [],
                      'mean_deficit': [], 'std_deficit': [], 'mean_norm_deficit': [], 'std_norm_deficit': []}
        data_dict_ = {'downlink_data': copy.deepcopy(data_dict_), 'uplink_data': copy.deepcopy(data_dict_)}  # replicating for uplink and downlink
    else:

        if data_[0]['downlink_results'] is not None:
            downlink_data = [x['downlink_results'] for x in data_]
            data_dict_['downlink_data'] = organize_data_matrix(data_=downlink_data,
                                                               data_dict_=data_dict_['downlink_data'], n_cells=n_cells,
                                                               n_samples=n_samples, n_centers=n_centers,
                                                               dist_typ=dist_typ)


        if data_[0]['uplink_results'] is not None:
            uplink_data = [x['uplink_results'] for x in data_]
            data_dict_['uplink_data'] = organize_data_matrix(data_=uplink_data,
                                                             data_dict_=data_dict_['uplink_data'], n_cells=n_cells,
                                                             n_samples=n_samples, n_centers=n_centers,
                                                             dist_typ=dist_typ)


    return data_dict_


def organize_data_matrix(data_, data_dict_, n_cells, n_samples, n_centers, dist_typ):
    # this funciton will take the simulation matrix data and will fit into the standard dictionary simulation output
    snr_cap_stats = [x['snr_cap_stats'] for x in data_]
    raw_data = [x['raw_data_dict'] for x in data_]
    if dist_typ == 'uniform':
        n_centers=1

    # saving cumulative simple metrics
    snr_cap_stats = np.array(snr_cap_stats)

    data_dict_['BSs'].append(n_cells)
    data_dict_['UEs'].append(n_samples * n_centers)
    data_dict_['mean_snr'].append(np.mean([x['mean_snr'] for x in snr_cap_stats]))
    data_dict_['std_snr'].append(np.mean([x['std_snr'] for x in snr_cap_stats]))
    data_dict_['mean_cap'].append(np.mean([x['mean_cap'] for x in snr_cap_stats]))
    data_dict_['std_cap'].append(np.mean([x['std_cap'] for x in snr_cap_stats]))
    data_dict_['mean_user_time'].append(np.mean([x['mean_user_time'] for x in snr_cap_stats]))
    data_dict_['std_user_time'].append(np.mean([x['std_user_time'] for x in snr_cap_stats]))
    data_dict_['mean_user_bw'].append(np.mean([x['mean_user_bw'] for x in snr_cap_stats]))
    data_dict_['std_user_bw'].append(np.mean([x['std_user_bw'] for x in snr_cap_stats]))

    if 'total_meet_criteria' in snr_cap_stats[0]:  # in the case of not using the capacity criteria
        data_dict_['total_meet_criteria'].append(np.mean([x['total_meet_criteria'] for x in snr_cap_stats]))
        data_dict_['mean_deficit'].append(np.mean([x['mean_deficit'] for x in snr_cap_stats]))
        data_dict_['std_deficit'].append(np.mean([x['std_deficit'] for x in snr_cap_stats]))
        data_dict_['mean_norm_deficit'].append(np.mean([x['mean_norm_deficit'] for x in snr_cap_stats]))
        data_dict_['std_norm_deficit'].append(np.mean([x['std_norm_deficit'] for x in snr_cap_stats]))

    # storing the raw data
    data_dict_['raw_data'].append(raw_data)

    return data_dict_


def create_subfolder(name_file, n_index, dict_name):
    # this function creates a folder inside the simulation one (name_file) to store data
    if not os.path.isdir('output'):
        os.mkdir('output')
    folder = os.path.dirname(__file__)
    folder = folder.replace('/', '\\')
    folder = '\\'.join(folder.split('\\')[:-1])
    folder += '\\output\\' + name_file + '\\'

    # creating subfolder
    folder = convert_file_path_os(folder + '\\' + str(n_index) + ' ' + dict_name + '\\')
    if not os.path.exists(folder):
        os.mkdir(folder)

    return folder

def load_data(name_file, return_path=False):
    # this function will load a dictionary in a pickle file in the provided path folder
    folder = os.path.dirname(__file__)
    folder = folder.replace('/', '\\')
    folder = '\\'.join(folder.split('\\')[:-1])
    folder += '\\output\\' + name_file + '\\'

    folder = convert_file_path_os(folder)

    path = folder
    folder += name_file + '.pkl'

    try:
        with open(folder, 'rb') as f:
            data_dict = pickle.load(f)
            f.close()
    except:
        return False

    if return_path:
        return data_dict[0], path
    else:
        return data_dict[0]

def save_data(path = None, data_dict = None):
    # this function will save a dictionary in a pickle file in the provided path folder
    if not path:
        folder = os.path.dirname(__file__)
        folder = folder.replace('/', '\\')
        folder = '\\'.join(folder.split('\\')[:-1])
        date = datetime.datetime.now()
        name_file = date.strftime('%x') + '-' + date.strftime('%X')
        name_file = name_file.replace('/', '_').replace(':', '_')
        folder += '\\output\\' + name_file + '\\'
        path = folder + name_file + '.pkl'

        path = convert_file_path_os(path)
        folder = convert_file_path_os(folder)
        print(folder)
        if os.path.exists(folder):
            os.mkdir(folder)
        else:
            os.makedirs(folder)

        return path, folder, name_file

    else:
        if data_dict and type(data_dict) is dict:
            with open(path, 'wb') as f:
                pickle.dump([data_dict], f)
                f.close()
                logging.info('Saved/updated file ' + path)
        else:
            logging.error('data_dictionary not provided!!!!')

def extract_parameter_from_raw(raw_data, parameter_name, data_index, calc=None, concatenate=True):
    # this function will pick the dictionary data and will organize and return the data for a specific parameter
    if calc is None:
        if concatenate:
            extracted_data = np.concatenate([x[parameter_name] for x in raw_data[data_index]])
        else:
            extracted_data = [x[parameter_name] for x in raw_data[data_index]]
    if calc == 'avg':
        extracted_data = [x[parameter_name].mean() for x in raw_data[data_index]]
    if calc == 'std':
        extracted_data = [x[parameter_name].std() for x in raw_data[data_index]]
    return extracted_data


def group_ue(data_dict, iter_dict_name, data_index=None):
    # this function will pick the output simulation data dict and will group the UEs by beam and sector and
    # also indicates the UEs that was not connected to the network
    dict = []
    if data_index is None:
        # bs_list = range(data_dict['BSs'].__len__())
        iter_list = range(data_dict[iter_dict_name].__len__())
    else:
        iter_list = [data_index]

    for bs_data_index in iter_list:
        nactive_ue_cnt = []  # UEs non-connected to the network
        ue_per_beam = []  # ues grouped by beam
        ue_per_sector = []  # ues grouped by sector
        active_ues = []  # UEs connected to the network

        ue_bs_tables = [x['ue_bs_table'] for x in data_dict['raw_data'][bs_data_index]]
        for i, ue_bs_tb in enumerate(ue_bs_tables):
            beam_comb = np.array(list(product(ue_bs_tb['bs_index'].unique(), ue_bs_tb['beam_index'].unique(), ue_bs_tb['sector_index'].unique())))
            sec_comb = np.array(list(product(ue_bs_tb['bs_index'].unique() , ue_bs_tb['sector_index'].unique())))
            act_beams = beam_comb[(beam_comb[:, 0] != -1) & (beam_comb[:, 1] != - 1) & (beam_comb[:, 2] != -1)]
            act_sec = sec_comb[(sec_comb[:, 0] != -1) & (sec_comb[:, 1] != - 1)]
            nactive_ue_cnt.append(np.sum(ue_bs_tb['bs_index'] == -1))
            active_ues.append(np.where(ue_bs_tb['bs_index'] != -1)[0])
            ue_bs_tb = np.array(ue_bs_tb)
            dummy_beam = []
            dummy_sec = []
            for index in act_beams:
                dummy_beam.append(np.where((ue_bs_tb[:, 0] == index[0]) & (ue_bs_tb[:, 1] == index[1]) &
                                            (ue_bs_tb[:, 2] == index[2]))[0])
            ue_per_beam.append(dummy_beam)
            for index in act_sec:
                dummy_sec.append(np.where((ue_bs_tb[:, 0] == index[0]) & (ue_bs_tb[:, 2] == index[1]))[0])
            ue_per_sector.append(dummy_sec)

        dict.append({'nactive_ue_cnt': nactive_ue_cnt, 'active_ues': active_ues, 'ue_per_beam':ue_per_beam,
                    'ue_per_sector': ue_per_sector})

    return dict


def ue_relative_index(data_dict, data_index=None):
    # this function will convert a output reference and return the relative index inside the dictionary
    if data_index is None:
        iter_list, _ = range(data_dict['BSs'].__len__())
    else:
        iter_list = [data_index]

    rel_index_tables = []

    for data_index in iter_list:
        ue_bs_tables = [x['ue_bs_table'] for x in data_dict['raw_data'][data_index]]
        nbs_rel_index = []
        for i, ue_bs_tb in enumerate(ue_bs_tables):
            a = np.where(ue_bs_tb['bs_index'] != -1)[0]
            b = np.where(a == a)[0]
            nbs_rel_index.append(pd.DataFrame(data=np.array([a, b]).T, columns=['ue_bs_index', 'relative_index']))
        rel_index_tables.append(nbs_rel_index)

    return rel_index_tables

def temp_data_save(zero_state=False, dict=None, batch_file=None):
    # to delete the temporary .pkl files inside the temp folder
    path = 'temp'
    if not os.path.isdir(path):
        os.mkdir(path)
    if zero_state:
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
    if dict is not None:
        pass
    if batch_file is not None:
        index = batch_file['index']
        data = batch_file['data']
        file_path = path + '/batch' + str(index) + '.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            f.close()
            logging.info('Saved/updated file ' + file_path)

def temp_data_load():
    # to load the temporary .pkl files inside the temp folder
    path = 'temp'
    data = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.find('batch') != -1:
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        data_ = pickle.load(f)
                    data = data + data_
        return data

    else:
        print('temp folder not located!!!')

def temp_data_delete(type):
    # to delete the temporary .pkl files inside the temp folder
    path = 'temp'
    if not ((type == 'batch') or (type == 'dict')):
        raise NameError('type not defined in temp_data_delete')

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.find(type) != -1:
                file_path = os.path.join(path, filename)
                os.remove(file_path)
