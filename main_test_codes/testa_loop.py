import matplotlib.pyplot as plt
from make_grid import Grid
from base_station import BaseStation
from macel import Macel
from antennas.antenna import ITU1336
import numpy as np
import time

if __name__ == '__main__':
    grid = Grid()  # grid object
    grid.make_grid(500, 500)  # creating a grid with x, y dimensions
    grid.make_points(dist_type='gaussian', samples=500, n_centers=4, random_centers=False, plot=False)  # distributing points aring centers in the grid

    # plt.matshow(grid.grid, origin='lower')
    # plt.title('grid samples')
    # plt.colorbar()
    # plt.show()

    antenna = ITU1336(gain=10, frequency=20, hor_beamwidth=10, ver_beamwidth=10)
    bs = BaseStation(frequency=3.5, tx_power=10, tx_height=30, bw=20, n_sectors=3, antenna=antenna, gain=10, downtilts=0, plot=False)
    cell_size = 30
    macel = Macel(grid=grid, n_centers=4, prop_model='FS', criteria=5E6, cell_size=cell_size, base_station=bs, log=False)
    macel.set_rx(rx_height=1.5)
    # macel.adjust_cell_number(min_n_cell=2)
    # macel.adjust_cell_number(min_n_cell=2, max_n_cell=2, max_iter=100, default_cells=np.array([100,100], [299, 87]))
    macel.adjust_cell_number(min_n_cell=4, max_n_cell=4, max_iter=100000)