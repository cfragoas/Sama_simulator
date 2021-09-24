import datetime
import logging
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import numpy as np
import tqdm
import multiprocessing, os
from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from make_grid import Grid
from base_station import BaseStation
from prop_models import generate_azimuth_map, generate_elevation_map, generate_euclidian_distance, generate_bf_gain
from clustering import Cluster
from macel import Macel
from user_eq import User_eq

# beamforming array diagram plot
# element = Element_ITU1336(max_gain=5, phi_3db=65,theta_3db=65, freq=10, plot=True)  # not using 1336 for now

# element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
# beamforming = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
#                                   vertical_spacing=0.5, point_theta=[0, -10, 15, 5], point_phi=[0, 10, 30, 40])
# beamforming.calculate_pattern(plot=True)


# testing a base station with beamforming

def macel_test(args):
    # extracting variables from pool call
    n_centers = args[0]
    bs = args[1]

    grid = Grid()  # grid object
    grid.make_grid(1000, 1000)  # creating a grid with x, y dimensions
    grid.make_points(dist_type='gaussian', samples=50, n_centers=4, random_centers=False, plot=False)  # distributing points aring centers in the grid
    ue = User_eq(positions=grid.grid, height=1.5)  #creating the user equipament object
    # element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
    # beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
    #                                   vertical_spacing=0.5)
    # bs = BaseStation(frequency=3.5, tx_power=50, tx_height=30, bw=300, n_sectors=3, antenna=beam_ant, gain=10, downtilts=0,
    #                  plot=False)
    # bs.sector_beam_pointing_configuration(n_beams=10)  # configuring the base stations to use 10 beams each
    cluster = Cluster()
    cluster.k_means(grid=grid.grid, n_clusters=n_centers)
    lines = grid.lines
    columns = grid.columns
    az_map = generate_azimuth_map(lines=lines, columns=columns, centroids=cluster.centroids,samples=cluster.features)
    dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=cluster.centroids, samples=cluster.features, plot=False)
    elev_map = generate_elevation_map(htx=30, hrx=1.5, d_euclid=dist_map, cell_size=100, samples=None)
    bs.beam_configuration(az_map=bs.beams_pointing)  # creating a beamforming configuration pointing to the the az_map points
    # bs.beam_configuration(az_map=az_map[0], elev_map=elev_map[0])  # rever essa parada aqui!!!
    # base_station_list = [bs] # creating a list is this case because theres is only one BS
    # gain_map = generate_bf_gain(elevation_map=elev_map, azimuth_map=az_map, base_station_list=base_station_list, sector_index=0)

    macel = Macel(grid=grid, n_centers=n_centers, prop_model='free space', criteria=0, cell_size=30, base_station=bs)
    macel.generate_base_station_list()
    macel.set_ue(ue=ue)
    ch_gain_map, sector_map = macel.generate_bf_gain_maps(az_map=az_map, elev_map=elev_map, dist_map=dist_map)
    macel.ue.acquire_bs_and_beam(ch_gain_map=ch_gain_map, sector_map=sector_map)  # calculating the best ch gain for each UE
    # macel.simulate_ue_bs_comm(simulation_time=1, time_slot=1)
    macel.send_ue_to_bs(simulation_time=1000, time_slot=1)
    snr_cap_stats = macel.simulate_ue_bs_comm(ch_gain_map=ch_gain_map)

    return(snr_cap_stats)

def load_data(name_file):
    folder = os.path.dirname(__file__)
    folder = '\\'.join(folder.split('\\')[:-1])
    folder += '\\output\\'
    folder += name_file

    with open(folder, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()

    return(data_dict)

def save_data(path = None, data_dict = None):
    if not path:
        folder = os.path.dirname(__file__)
        folder = '\\'.join(folder.split('\\')[:-1])
        folder += '\\output\\'
        date = datetime.datetime.now()
        name_file = date.strftime('%x') + '-' + date.strftime('%X') + '.pkl'
        name_file = name_file.replace('/', '_').replace(':', '_')
        path = folder + name_file

        return path

    else:
        if data_dict and type(data_dict) is dict:
            with open(path, 'wb') as f:
                pickle.dump([data_dict], f)
                f.close()
                logging.info('Saved/updated file ' + path)
        else:
            logging.error('data_dictionary not provided!!!!')

def plot(mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw, std_user_bw):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)
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
    plt.show()


if __name__ == '__main__':
    # classes that don't change in the loop
    element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
    beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
                                   vertical_spacing=0.5)
    base_station = BaseStation(frequency=3.5, tx_power=50, tx_height=30, bw=300, n_sectors=3, antenna=beam_ant, gain=10,
                     downtilts=0, plot=False)
    base_station.sector_beam_pointing_configuration(n_beams=10)

    threads = os.cpu_count()
    if threads > 61:  # to run in processors with 30+ cores
        threads = 61
    p = multiprocessing.Pool(processes=threads-1)

    data_dict = {}
    data_dict['BSs'] = 0
    data_dict['mean_snr'] = []
    data_dict['std_snr'] = []
    data_dict['mean_cap'] = []
    data_dict['std_cap'] = []
    data_dict['mean_user_time'] = []
    data_dict['std_user_time'] = []
    data_dict['mean_user_bw'] = []
    data_dict['std_user_bw'] = []

    path = save_data()  # storing the path used to save in all iterations

    max_iter = 100
    for n_cells in range(1, 25):
        print('running with ', n_cells,' BSs')
        data = list(
                    tqdm.tqdm(p.imap_unordered(macel_test, [(n_cells, base_station) for i in range(max_iter)]), total=max_iter
                ))
        print(np.mean(data[0]))
        print(os.linesep)

        data = np.array(data)

        data_dict['BSs'] = n_cells
        data_dict['mean_snr'].append(np.mean(data[:, 0]))
        data_dict['std_snr'].append(np.mean(data[:, 1]))
        data_dict['mean_cap'].append(np.mean(data[:, 2]))
        data_dict['std_cap'].append(np.mean(data[:, 3]))
        data_dict['mean_user_time'].append(np.mean(data[:, 4]))
        data_dict['std_user_time'].append(np.mean(data[:, 5]))
        data_dict['mean_user_bw'].append(np.mean(data[:, 6]))
        data_dict['std_user_bw'].append(np.mean(data[:, 7]))

        save_data(path=path, data_dict=data_dict) # saving/updating data

    plot(mean_snr=data_dict['mean_snr'], std_snr=data_dict['std_snr'], mean_cap=data_dict['mean_cap'], std_cap=data_dict['std_cap'],
         mean_user_time=data_dict['mean_user_time'], std_user_time=data_dict['std_user_time'], mean_user_bw=data_dict['mean_user_bw'],
         std_user_bw=data_dict['std_user_bw'])


