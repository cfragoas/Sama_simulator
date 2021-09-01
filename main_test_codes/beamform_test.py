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
grid = Grid()  # grid object
grid.make_grid(1000, 1000)  # creating a grid with x, y dimensions
grid.make_points(dist_type='gaussian', samples=10, n_centers=4, random_centers=False, plot=False)  # distributing points aring centers in the grid
ue = User_eq(positions=grid.grid, height=1.5)  #creating the user equipament object
element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
                                  vertical_spacing=0.5)
bs = BaseStation(frequency=3.5, tx_power=10, tx_height=30, bw=20, n_sectors=3, antenna=beam_ant, gain=10, downtilts=0,
                 plot=False)
bs.sector_beam_pointing_configuration(n_beams=10)  # configuring the base stations to use 10 beams each
cluster = Cluster()
cluster.k_means(grid=grid.grid, n_clusters=2)
lines = grid.lines
columns = grid.columns
az_map = generate_azimuth_map(lines=lines, columns=columns, centroids=cluster.centroids,samples=cluster.features)
dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=cluster.centroids, samples=cluster.features, plot=False)
elev_map = generate_elevation_map(htx=30, hrx=1.5, d_euclid=dist_map, cell_size=30, samples=None)
bs.beam_configuration(az_map=bs.beams_pointing)  # creating a beamforming configuration pointing to the the az_map points
# bs.beam_configuration(az_map=az_map[0], elev_map=elev_map[0])  # rever essa parada aqui!!!
# base_station_list = [bs] # creating a list is this case because theres is only one BS
# gain_map = generate_bf_gain(elevation_map=elev_map, azimuth_map=az_map, base_station_list=base_station_list, sector_index=0)

macel = Macel(grid=grid, n_centers=2, prop_model='free space', criteria=0, cell_size=30, base_station=bs)
macel.generate_base_station_list()
macel.set_ue(ue=ue)
ch_gain_map, sector_map = macel.generate_bf_gain_maps(az_map=az_map, elev_map=elev_map, dist_map=dist_map)
macel.ue.acquire_bs_and_beam(ch_gain_map=ch_gain_map, sector_map=sector_map)  # calculating the best ch gain for each UE
# macel.simulate_ue_bs_comm(simulation_time=1, time_slot=1)
macel.send_ue_to_bs(simulation_time=1000, time_slot=1)
# for base_station in macel.base_station_list:
#     base_station.generate_beam_timing()
print('ui')