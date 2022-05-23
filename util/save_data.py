import os, datetime, platform, logging
import numpy as np
import pickle

def write_conf(folder, parameters, yml_file=None):
    with open(folder + 'configuration.txt', 'w') as f:
        for key, item in parameters.items():
            f.writelines(str(key) + ': ' + str(item) + '\n')

    if yml_file is not None:
        pass
    # save param file here



def macel_data_dict(data_dict_=None, data_=None, n_cells=None):
    if not data_ or not data_dict_:
        data_dict_ = {'BSs': 0, 'mean_snr': [], 'std_snr': [], 'mean_cap': [], 'std_cap': [], 'mean_user_time': [],
                      'std_user_time': [], 'mean_user_bw': [], 'std_user_bw': [], 'raw_data': [], 'total_meet_criteria': [],
                      'mean_deficit': [], 'std_deficit': [], 'mean_norm_deficit': [], 'std_norm_deficit': []}
    else:
        snr_cap_stats = [x[0] for x in data_]
        raw_data = [x[1] for x in data_]


        # saving cumulative simple metrics
        snr_cap_stats = np.array(snr_cap_stats)

        data_dict_['BSs'] = n_cells
        data_dict_['mean_snr'].append(np.mean(snr_cap_stats[:, 0]))
        data_dict_['std_snr'].append(np.mean(snr_cap_stats[:, 1]))
        data_dict_['mean_cap'].append(np.mean(snr_cap_stats[:, 2]))
        data_dict_['std_cap'].append(np.mean(snr_cap_stats[:, 3]))
        data_dict_['mean_user_time'].append(np.mean(snr_cap_stats[:, 4]))
        data_dict_['std_user_time'].append(np.mean(snr_cap_stats[:, 5]))
        data_dict_['mean_user_bw'].append(np.mean(snr_cap_stats[:, 6]))
        data_dict_['std_user_bw'].append(np.mean(snr_cap_stats[:, 7]))
        data_dict_['total_meet_criteria'].append(np.mean(snr_cap_stats[:, 8]))
        data_dict_['mean_deficit'].append(np.mean(snr_cap_stats[:, 9]))
        data_dict_['std_deficit'].append(np.mean(snr_cap_stats[:, 10]))
        data_dict_['mean_norm_deficit'].append(np.mean(snr_cap_stats[:, 11]))
        data_dict_['std_norm_deficit'].append(np.mean(snr_cap_stats[:, 12]))

        # saving the raw data
        data_dict_['raw_data'].append(raw_data)

    return data_dict_

def load_data(name_file):
    folder = os.path.dirname(__file__)
    folder = '\\'.join(folder.split('\\')[:-1])
    folder += '\\output\\' + name_file + '\\'
    folder += name_file + '.pkl'

    with open(folder, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()

    return(data_dict)

def save_data(path = None, data_dict = None):
    if not path:
        folder = os.path.dirname(__file__)
        folder = folder.replace('/', '\\')
        folder = '\\'.join(folder.split('\\')[:-1])
        date = datetime.datetime.now()
        name_file = date.strftime('%x') + '-' + date.strftime('%X')
        name_file = name_file.replace('/', '_').replace(':', '_')
        folder += '\\output\\' + name_file + '\\'
        path = folder + name_file + '.pkl'

        if platform.system() == 'Darwin':
            path = path.replace('\\', '/')
            folder = folder.replace('\\', '/')
        print(folder)
        os.mkdir(folder)

        return path, folder

    else:
        if data_dict and type(data_dict) is dict:
            with open(path, 'wb') as f:
                pickle.dump([data_dict], f)
                f.close()
                logging.info('Saved/updated file ' + path)
        else:
            logging.error('data_dictionary not provided!!!!')