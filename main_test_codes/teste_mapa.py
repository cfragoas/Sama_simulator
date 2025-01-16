import matplotlib.pyplot as plt
from map import Map
from make_grid import Grid
from macel import Macel
from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from base_station import BaseStation

# testing the integration of maps with the optimization algorithm

def convert_file_path_os(path):
    import platform
    if platform.system() == 'Darwin':
        path = path.replace('\\', '/')
    return path


folder = '..\\map_data'
mapa = Map()
mapa.load(path=convert_file_path_os(folder + '\\30m.pkl'))
mapa.load_general_map_info(path=convert_file_path_os(folder + '\\Brasil_Sce_2010.csv'),
                           id_column='Cod_Setor', delimiter=';')
idx_map, mask = mapa.clip_shape(shape=mapa.idx_mtx, criteria='Tijuca', var='Nm_Bairro',
                                save=True, plot=True)
wgt_map = mapa.apply_mask(shape=mapa.wgt_mtx, mask=mask, plot=True)
dst_map = mapa.apply_mask(shape=mapa.dst_mtx, mask=mask, plot=True)
mapa.generate_samples(n_samples=1000)
mapa_grid = mapa.make_grid()
# points_map, point_list = mapa.uniform_dist(n_samples=1000, id_mtx=idx_map, dnst_map=dst_map)


# teste_grid = Grid()
# teste_grid.make_grid(500, 500)
# teste_grid.make_points(dist_type='gaussian', samples=200, n_centers=4, random_centers=False,
#                           plot=False)  # distributing points around centers in the grid

# testing with Macel class
element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
                               vertical_spacing=0.5)
base_station = BaseStation(frequency=3.5, tx_power=20, tx_height=30, bw=300, n_sectors=3, antenna=beam_ant, gain=10,
                 downtilts=0, plot=False)
base_station.sector_beam_pointing_configuration(n_beams=10)

macel = Macel(grid=mapa_grid, prop_model='free space', criteria=50,
                  cell_size=30, base_station=base_station,
                  simulation_time=1000, scheduling_opt=False, simplified_schdl=False)
macel.set_ue(hrx=1.5)

snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=4, output_typ='complete', clustering=True)
import numpy as np
plt.imshow(mapa.sample_mtx)
plt.colorbar()
for point in raw_data['position'][0]:
    plt.scatter(point[1], point[0])
plt.show()
