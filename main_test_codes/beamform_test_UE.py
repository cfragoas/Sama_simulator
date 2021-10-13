import copy
import datetime, pickle, os, logging, multiprocessing, tqdm
import matplotlib.pyplot as plt
import numpy as np

from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from base_station import BaseStation
from make_grid import Grid
from macel import Macel

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
                      'std_user_time': [], 'mean_user_bw': [], 'std_user_bw': [], 'raw_data': []}
    else:
        snr_cap_stats = [x[0] for x in data]
        raw_data = [x[1] for x in data]


        # saving cumulative simple metrics
        snr_cap_stats = np.array(snr_cap_stats)

        data_dict['BSs'] = n_cells
        data_dict['mean_snr'].append(np.mean(snr_cap_stats[:, 0]))
        data_dict['std_snr'].append(np.mean(snr_cap_stats[:, 1]))
        data_dict['mean_cap'].append(np.mean(snr_cap_stats[:, 2]))
        data_dict['std_cap'].append(np.mean(snr_cap_stats[:, 3]))
        data_dict['mean_user_time'].append(np.mean(snr_cap_stats[:, 4]))
        data_dict['std_user_time'].append(np.mean(snr_cap_stats[:, 5]))
        data_dict['mean_user_bw'].append(np.mean(snr_cap_stats[:, 6]))
        data_dict['std_user_bw'].append(np.mean(snr_cap_stats[:, 7]))
        data_dict['meet_criteria'].append(snr_cap_stats[:, 8])

        # saving the raw data
        data_dict['raw_data'].append(raw_data)

    return data_dict_


def plot_curves(mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw, std_user_bw,
         meet_criteria, max_iter, n_bs, n_ue_vec, individual=False, path=''):
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


    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, dpi=100, figsize=(50, 30))
    fig.suptitle('Metrics evolution by UE number - ' + str(max_iter) + ' iterations and ' + str(n_bs) + ' BSs')
    ax1.plot(n_ue_vec, mean_snr)
    ax1.set_title('Mean SNIR')
    ax2.plot(n_ue_vec, std_snr)
    ax2.set_title('std SNIR')
    ax3.plot(n_ue_vec, mean_cap)
    ax3.set_title('Mean Capacity (Mbps)')
    ax4.plot(n_ue_vec, std_cap)
    ax4.set_title('std Capacity (Mbps)')
    ax5.plot(n_ue_vec, mean_user_time)
    ax5.set_title('Mean user time (s)')
    ax6.plot(n_ue_vec, std_user_time)
    ax6.set_title('std user time (s)')
    ax7.plot(n_ue_vec, mean_user_bw)
    ax7.set_title('Mean user bw (MHz)')
    ax8.plot(n_ue_vec, std_user_bw)
    ax8.set_title('std user bw (MHz)')
    plt.rcParams['font.size'] = '10'
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig(path + 'perf_curves.png')

    plt.close('all')

    plt.plot(n_ue_vec, meet_criteria)
    plt.title('% of UE that meets the ' + str(criteria) + ' Mbps criteria')
    plt.savefig(path + 'meet_criteria.png')
    plt.close('all')

def plot_hist(raw_data, path, n_bs, n_ue):
    #creating subfolder
    path = path + '\\' + str(n_ue) + 'UEs\\'
    if not os.path.exists(path):
        os.mkdir(path)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=100, figsize=(50, 30))
    fig.suptitle('Metrics using ' + str(n_bs) + ' BSs and ' + str(n_ue) + 'UEs -' + str(max_iter) + ' iterations')

    # SNR
    snr = np.concatenate([x['snr'] for x in raw_data])
    ax1.hist(snr, bins=100)
    ax1.set_title('SNIR (dB)')
    f1 = plt.figure(8, dpi=100)
    plt.hist(snr, bins=100)
    plt.title('SNIR (dB)')
    # plt.show()
    plt.savefig(path + 'snr_' + str(n_ue) + ' UE.png')

    # CAP
    cap = np.concatenate([x['cap'] for x in raw_data])
    ax2.hist(cap, bins=1000, range=(0, 120))
    ax2.set_title('Throughput (Mbps)')
    f2 = plt.figure(3, dpi=150)
    plt.hist(cap, bins=1000,  range=(0, 120))
    plt.title('Throughput (Mbps)')
    # plt.show()
    plt.savefig(path + 'cap_' + str(n_ue) + ' UE.png')

    # Users p/ BS
    user_bs = np.concatenate([x['user_bs'] for x in raw_data])
    ax3.hist(user_bs, bins=100)
    ax3.set_title('UEs per BS')
    f3 = plt.figure(4, dpi=150)
    plt.hist(user_bs, bins=100)
    plt.title('Number of UEs per BS')
    # plt.show()
    plt.savefig(path + 'user_bs_' + str(n_ue) + ' UE.png')

    # Number of active beams
    act_beams = np.concatenate([x['act_beams'] for x in raw_data])
    ax4.hist(act_beams, bins=11)
    ax4.set_title('Act beam p/BS')
    f4 = plt.figure(5, dpi=150)
    plt.hist(act_beams, bins=11)
    plt.title('Number of Active beams per BS')
    # plt.show()
    plt.savefig(path + 'act_beams_' + str(n_ue) + ' UE.png')

    # time per user in 1s
    user_time = np.concatenate([x['user_time'] for x in raw_data])
    ax5.hist(user_time, bins=50)
    ax5.set_title('UE time in 1s')
    f5 = plt.figure(6, dpi=150)
    plt.hist(user_time, bins=50)
    plt.title('UE active time in 1s')
    # plt.show()
    plt.savefig(path + 'user_time_' + str(n_ue) + ' UE.png')

    # bandwidth per user
    user_bw = np.concatenate([x['user_bw'] for x in raw_data])
    ax6.hist(user_bw, bins=20)
    ax6.set_title('BW p/ UE(MHz)')
    f6 = plt.figure(7, dpi=150)
    plt.hist(user_bw, bins=20)
    plt.title('Bandwidth per UE (MHz)')
    # plt.show()
    plt.savefig(path + 'bw_user_' + str(n_ue) + ' UE.png')

    plt.rcParams['font.size'] = '10'
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + 'Metrics_' + str(n_ue) + ' UE.png')

    plt.close('all')

def plot_surface(grid, position, parameter, path, n_bs, n_ue):
    # creating subfolder
    path = path + '\\' + str(n_ue) + 'UEs\\'
    if not os.path.exists(path):
        os.mkdir(path)

    snr_sum = grid
    counter = copy.deepcopy(grid)
    # plot surface
    X = position[:, :, 0]
    Y = position[:, :, 1]
    #X, Y = np.meshgrid(X, Y)

    for i, [x, y] in enumerate(zip(X, Y)):
        for j, [k, l] in enumerate(zip(x,y)):
            snr_sum[k, l] += parameter[i]
            counter[k, l] += 1

    mean_snr = snr_sum/np.where(counter == 0, 1, counter)

    fig1, ax1 = plt.subplots(1, dpi=300)
    fig1.suptitle('Accumulated SNIR ' + str(n_bs) + ' BSs and ' + str(n_ue) + 'UEs -' + str(max_iter) + ' iterations')
    z = ax1.matshow(snr_sum, origin='lower')
    fig1.colorbar(z, ax=ax1)

    plt.savefig(path + 'accum_cap_surf_' + str(n_ue) + ' UE.png')
    plt.close('all')

    fig2, ax1 = plt.subplots(1, dpi=300)
    fig2.suptitle('Average SNIR ' + str(n_bs) + ' BSs and ' + str(n_ue) + 'UEs -' + str(max_iter) + ' iterations')
    z = ax1.matshow(mean_snr, origin='lower')
    fig1.colorbar(z, ax=ax1)
    print(parameter.shape[0])
    plt.savefig(path + 'mean_cap_surf_' + str(n_ue) + ' UE.png')

    plt.close('all')

    fig3, ax1 = plt.subplots(1, dpi=300)
    fig3.suptitle('BS distribution ' + str(n_bs) + ' BSs and ' + str(n_ue) + 'UEs -' + str(max_iter) + ' iterations')
    z = ax1.matshow(counter, origin='lower')
    fig3.colorbar(z, ax=ax1)

    plt.savefig(path + 'bs_dist_surf_' + str(n_ue) + ' UE.png')
    plt.close('all')


def simulate_ue_macel (args):
    n_bs = args[0]
    macel = args[1]
    samples = args[2]

    macel.grid.make_points(dist_type='gaussian', samples=samples, n_centers=4, random_centers=False,
                          plot=False)  # distributing points around centers in the grid
    macel.set_ue(hrx=1.5)
    snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=n_bs, output_typ='complete')

    return(snr_cap_stats, raw_data)


if __name__ == '__main__':
    # parameters
    criteria = 50
    n_centers = 4 # ARRUMAR ISSO AQUI!!!!!
    n_cells = 4
    max_iter = 100
    min_samples = 10
    max_samples = 300
    step = 20
    threads = None

    sample_vec = []

    if threads == None:
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
    macel = Macel(grid=grid, prop_model='free space', criteria=criteria, cell_size=30, base_station=base_station)

    for samples in range(min_samples, max_samples, step):
        macel.grid.clear_grid()  # added to avoid increasing UE number without intention
        sample_vec.append(samples * n_centers)
        print('running with ', samples * n_centers,' UEs')
        data = list(
                    tqdm.tqdm(p.imap_unordered(simulate_ue_macel, [(n_cells, macel, samples) for i in range(max_iter)]), total=max_iter
                ))
        snr_cap_stats = [x[0] for x in data]
        raw_data = [x[1] for x in data]

        print('Mean SNR:', np.mean(snr_cap_stats[0]), ' dB')
        print('Mean cap:', np.mean(snr_cap_stats[2]), ' Mbps')
        print(os.linesep)

        plot_hist(raw_data=raw_data, path=folder, n_bs=n_cells, n_ue=samples*n_centers)

        plot_surface(grid=grid.grid, position=np.concatenate([x['position'] for x in raw_data]),
                     parameter=np.array(snr_cap_stats)[:, 2], path=folder, n_bs=n_cells, n_ue=samples*n_centers)

        data_dict = macel_data_dict(data_dict_=data_dict, data_=data)

        save_data(path=path, data_dict=data_dict)  # saving/updating data

        plot_curves(mean_snr=data_dict['mean_snr'], std_snr=data_dict['std_snr'], mean_cap=data_dict['mean_cap'],
             std_cap=data_dict['std_cap'],
             mean_user_time=data_dict['mean_user_time'], std_user_time=data_dict['std_user_time'],
             mean_user_bw=data_dict['mean_user_bw'],
             std_user_bw=data_dict['std_user_bw'], meet_criteria=data_dict['meet_criteria'] ,max_iter=max_iter , n_bs=n_cells, n_ue_vec=sample_vec,individual=False, path=folder)
