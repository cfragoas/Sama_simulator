import datetime, pickle, os, logging, multiprocessing, tqdm
import matplotlib.pyplot as plt
import numpy as np

from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from base_station import BaseStation
from user_eq import User_eq
from make_grid import Grid
from macel2 import Macel

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

def macel_data_dict(data_dict_=None, data_=None):
    if not data_ or not data_dict_:
        data_dict_ = {'BSs': 0, 'mean_snr': [], 'std_snr': [], 'mean_cap': [], 'std_cap': [], 'mean_user_time': [],
                      'std_user_time': [], 'mean_user_bw': [], 'std_user_bw': []}
    else:
        data_ = np.array(data_)

        data_dict['BSs'] = n_cells
        data_dict['mean_snr'].append(np.mean(data_[:, 0]))
        data_dict['std_snr'].append(np.mean(data_[:, 1]))
        data_dict['mean_cap'].append(np.mean(data_[:, 2]))
        data_dict['std_cap'].append(np.mean(data_[:, 3]))
        data_dict['mean_user_time'].append(np.mean(data_[:, 4]))
        data_dict['std_user_time'].append(np.mean(data_[:, 5]))
        data_dict['mean_user_bw'].append(np.mean(data_[:, 6]))
        data_dict['std_user_bw'].append(np.mean(data_[:, 7]))

    return data_dict_


def plot_curve(mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw, std_user_bw,
         max_iter, individual=False, path=''):
    if individual:
        # Mean SNIR
        plt.plot(mean_snr)
        plt.title('Mean SNIR')
        plt.savefig()

        # std SNIR
        plt.plot(std_snr)
        plt.title('std SNIR')
        plt.savefig()

        # mean CAP
        plt.plot(mean_cap)
        plt.title('std SNIR')
        plt.savefig()

        # std CAP
        plt.plot(std_cap)
        plt.title('std SNIR')
        plt.savefig()

        # mean user time
        plt.plot(mean_user_time)
        plt.title('std SNIR')
        plt.savefig()

        # std user time
        plt.plot(std_user_time)
        plt.title('std SNIR')
        plt.savefig()

        # mean user bw
        plt.plot(mean_user_bw)
        plt.title('std SNIR')
        plt.savefig()

        # std user bw
        plt.plot(std_user_bw)
        plt.title('std SNIR')
        plt.savefig()


    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, dpi=500)
    fig.suptitle('Metrics evolution by BS number - ' + str(max_iter) + ' iterations')
    ax1.plot(mean_snr)
    ax1.set_title('Mean SNIR')
    ax2.plot(std_snr)
    ax2.set_title('std SNIR')
    ax3.plot(mean_cap)
    ax3.set_title('Mean Capacity (Mbps)')
    ax4.plot(std_cap)
    ax4.set_title('std Capacity (Mbps)')
    ax5.plot(mean_user_time)
    ax5.set_title('Mean user time (s)')
    ax6.plot(std_user_time)
    ax6.set_title('std user time (s)')
    ax7.plot(mean_user_bw)
    ax7.set_title('Mean user bw (MHz)')
    ax8.plot(std_user_bw)
    ax8.set_title('std user bw (MHz)')
    fig.tight_layout()
    # plt.show()
    plt.savefig(path + 'perf_curves.png')

def plot_hist(raw_data, path, n_bs):
    #creating subfolder
    path = path + '\\' + str(n_bs) + 'BSs\\'
    os.mkdir(path)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=300)
    fig.suptitle('Metrics using ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')

    # SNR
    snr = np.concatenate([x['snr'] for x in raw_data])
    ax1.hist(snr, bins=100)
    ax1.set_title('SNIR (dB)')
    f = plt.figure(2, dpi=100)
    plt.hist(snr, bins=100)
    plt.title('SNIR (dB)')
    # plt.show()
    plt.savefig(path + 'snr_' + str(n_bs) + ' BS.png')

    # CAP
    cap = np.concatenate([x['cap'] for x in raw_data])
    ax2.hist(cap, bins=100)
    ax2.set_title('Throughput (Mbps)')
    f = plt.figure(3, dpi=150)
    plt.hist(cap, bins=100)
    plt.title('Throughput (Mbps)')
    # plt.show()
    plt.savefig(path + 'cap_' + str(n_bs) + ' BS.png')

    # Users p/ BS
    user_bs = np.concatenate([x['user_bs'] for x in raw_data])
    ax3.hist(user_bs, bins=100)
    ax3.set_title('UEs per BS')
    f = plt.figure(4, dpi=150)
    plt.hist(user_bs, bins=100)
    plt.title('Number of UEs per BS')
    # plt.show()
    plt.savefig(path + 'user_bs_' + str(n_bs) + ' BS.png')

    # Number of active beams
    act_beams = np.concatenate([x['act_beams'] for x in raw_data])
    ax4.hist(act_beams, bins=11)
    ax4.set_title('Act beam p/BS')
    f = plt.figure(5, dpi=150)
    plt.hist(act_beams, bins=11)
    plt.title('Number of Active beams per BS')
    # plt.show()
    plt.savefig(path + 'act_beams_' + str(n_bs) + ' BS.png')

    # time per user in 1s
    user_time = np.concatenate([x['user_time'] for x in raw_data])
    ax5.hist(user_time, bins=50)
    ax5.set_title('UE time in 1s')
    f = plt.figure(6, dpi=150)
    plt.hist(user_time, bins=50)
    plt.title('UE active time in 1s')
    # plt.show()
    plt.savefig(path + 'user_time_' + str(n_bs) + ' BS.png')

    # bandwidth per user
    user_bw = np.concatenate([x['user_bw'] for x in raw_data])
    ax6.hist(user_bw, bins=20)
    ax6.set_title('BW p/ UE(MHz)')
    f = plt.figure(7, dpi=150)
    plt.hist(user_bw, bins=20)
    plt.title('Bandwidth per UE (MHz)')
    # plt.show()
    plt.savefig(path + 'bw_user_' + str(n_bs) + ' BS.png')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + 'Metrics_' + str(n_bs) + ' BS.png')


def simulate_ue_macel (args):
    n_bs = args[0]
    macel = args[1]

    macel.grid.make_points(dist_type='gaussian', samples=200, n_centers=4, random_centers=False,
                          plot=False)  # distributing points around centers in the grid
    macel.set_ue(hrx=1.5)
    snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=n_bs, output_typ='complete')

    return(snr_cap_stats, raw_data)


if __name__ == '__main__':
    # parameters
    samples = 100  # REDO - NAO FUNCIONAAAAAA
    max_iter = 10000
    min_bs = 1
    max_bs = 30

    threads = os.cpu_count()
    if threads > 61:  # to run in processors with 30+ cores
        threads = 61
    p = multiprocessing.Pool(processes=threads - 1)

    path, folder = save_data()  # storing the path used to save in all iterations

    data_dict = macel_data_dict()

    # ==========================================
    grid = Grid()  # grid object
    grid.make_grid(1000, 1000)

    element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
    beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
                                   vertical_spacing=0.5)
    base_station = BaseStation(frequency=3.5, tx_power=50, tx_height=30, bw=300, n_sectors=3, antenna=beam_ant, gain=10,
                     downtilts=0, plot=False)
    base_station.sector_beam_pointing_configuration(n_beams=10)
    macel = Macel(grid=grid, prop_model='free space', criteria=0, cell_size=30, base_station=base_station)

    for n_cells in range(min_bs, max_bs):
        print('running with ', n_cells,' BSs')
        data = list(
                    tqdm.tqdm(p.imap_unordered(simulate_ue_macel, [(n_cells, macel) for i in range(max_iter)]), total=max_iter
                ))
        snr_cap_stats = [x[0] for x in data]
        raw_data = [x[1] for x in data]

        print('Mean SNR:', np.mean(snr_cap_stats[0]), ' dB')
        print('Mean cap:', np.mean(snr_cap_stats[2]), ' Mbps')
        print(os.linesep)

        plot_hist(raw_data=raw_data, path=folder, n_bs=n_cells)

        data_dict = macel_data_dict(data_dict_=data_dict, data_=snr_cap_stats)

        save_data(path=path, data_dict=data_dict)  # saving/updating data

        plot_curve(mean_snr=data_dict['mean_snr'], std_snr=data_dict['std_snr'], mean_cap=data_dict['mean_cap'],
             std_cap=data_dict['std_cap'],
             mean_user_time=data_dict['mean_user_time'], std_user_time=data_dict['std_user_time'],
             mean_user_bw=data_dict['mean_user_bw'],
             std_user_bw=data_dict['std_user_bw'],max_iter=max_iter ,individual=False, path=folder)
