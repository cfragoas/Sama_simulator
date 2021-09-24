from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from base_station import BaseStation
from make_grid import Grid
from macel2 import Macel

grid = Grid()  # grid object
grid.make_grid(1000, 1000)

element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
                               vertical_spacing=0.5)
base_station = BaseStation(frequency=3.5, tx_power=50, tx_height=30, bw=300, n_sectors=3, antenna=beam_ant, gain=10,
                 downtilts=0, plot=False)
base_station.sector_beam_pointing_configuration(n_beams=10)
macel = Macel(grid=grid, prop_model='free space', criteria=0, cell_size=30, base_station=bs)