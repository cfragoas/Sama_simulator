import numpy as np
import matplotlib.pyplot as plt
import tqdm
import multiprocessing, os

from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from base_station import BaseStation
from make_grid import Grid
from macel import Macel
from demos_and_examples.kmeans_from_scratch import K_Means_XP

def simulate_ue_macel(args):
    n_bs = args[0]
    macel = args[1]
    samples = args[2]
    predetermined_centroids = args[3]

    macel.grid.make_points(dist_type='gaussian', samples=samples, n_centers=4, random_centers=False,
                           plot=False)  # distributing points around centers in the grid
    macel.set_ue(hrx=1.5)
    # kmeans = K_Means_XP(k=n_bs)
    # kmeans.fit(data=macel.grid, predetermined_centroids=predetermined_centroids)
    snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=n_bs, output_typ='complete',
                                                           predetermined_centroids=predetermined_centroids)


if __name__ == '__main__':
    # PARAMETERS
    criteria = 50  # capacity metric evaluation parameter (Mbps)
    samples = 100  # number of samples per gaussian center
    max_iter = 100  # number of iterations per BS/UE set
    min_bs = 4  # minimum number of BSs (if using predetermined centroids, needs to be more than that)
    max_bs = 30  # maximum number of BSs
    predetermined_centroids = np.array(([250, 250], [400, 800], [500, 100]))
    threads = None

    bs_vec = []

    if threads == None:
        threads = os.cpu_count()
    if threads > 61:  # to run in processors with 30+ cores
        threads = 61
    p = multiprocessing.Pool(processes=threads - 1)

    # path, folder = save_data()  # storing the path used to save in all iterations
    #
    # data_dict = macel_data_dict()

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

    for n_cells in range(min_bs, max_bs + 1):
        macel.grid.clear_grid()  # added to avoid increasing UE number without intention
        bs_vec.append(n_cells)
        print('running with ', n_cells,' BSs')
        data = list(
                    tqdm.tqdm(p.imap_unordered(simulate_ue_macel, [(n_cells, macel, samples, predetermined_centroids) for i in range(max_iter)]), total=max_iter
                ))
