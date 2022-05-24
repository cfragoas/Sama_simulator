
def simulate_macel_downlink(args):  # todo - fix the and check all the options here
    n_bs = args[0]
    macel = args[1]
    n_samples = args[2]
    n_centers = args[3]

    macel.grid.make_points(dist_type='gaussian', samples=n_samples, n_centers=n_centers, random_centers=False,
                           plot=False)  # distributing points around centers in the grid
    macel.set_ue(hrx=1.5)
    snr_cap_stats, raw_data = macel.place_and_configure_bs(n_centers=n_bs, output_typ='complete', clustering=True)
    # snr_cap_stats = macel.place_and_configure_bs(n_centers=n_bs, output_typ='simple', clustering=False)
    return(snr_cap_stats, raw_data)


def create_enviroment(parameters):
    from make_grid import Grid
    from antennas.ITU2101_Element import Element_ITU2101
    from antennas.beamforming import Beamforming_Antenna
    from base_station import BaseStation
    from macel import Macel


    grid = Grid()  # grid object
    grid.make_grid(lines=parameters['roi_param']['grid_lines'],
                   columns=parameters['roi_param']['grid_columns'])

    element = Element_ITU2101(max_gain=parameters['antenna_param']['max_element_gain'],
                              phi_3db=parameters['antenna_param']['phi_3db'],
                              theta_3db=parameters['antenna_param']['theta_3db'],
                              front_back_h=parameters['antenna_param']['front_back_h'],
                              sla_v=parameters['antenna_param']['sla_v'],
                              plot=False)

    beam_ant = Beamforming_Antenna(ant_element=element,
                                   frequency=None,
                                   n_rows=parameters['antenna_param']['n_rows'],
                                   n_columns=parameters['antenna_param']['n_columns'],
                                   horizontal_spacing=parameters['antenna_param']['horizontal_spacing'],
                                   vertical_spacing=parameters['antenna_param']['vertical_spacing'])

    base_station = BaseStation(frequency=parameters['bs_param']['freq'],
                               tx_power=parameters['bs_param']['tx_power'],
                               tx_height=parameters['bs_param']['htx'],
                               bw=parameters['bs_param']['bw'],
                               n_sectors=parameters['bs_param']['n_sectors'],
                               antenna=beam_ant,
                               gain=None,
                               downtilts=parameters['bs_param']['downtilt'],
                               plot=False)

    base_station.sector_beam_pointing_configuration(n_beams=parameters['bs_param']['n_beams'])

    macel = Macel(grid=grid, prop_model='free space',
                  criteria=parameters['macel_param']['criteria'],
                  cell_size=parameters['roi_param']['cel_size'],  # todo - ARRUMAR ISSO AQUI (passar para o grid)!!!
                  base_station=base_station,
                  simulation_time=parameters['macel_param']['time_slots'],
                  scheduling_opt=parameters['macel_param']['scheduling_opt'],
                  simplified_schdl=parameters['macel_param']['simplified_schdl'])

    return macel


def prep_multiproc(threads):
    import multiprocessing
    import os

    if threads == 0:
        threads = os.cpu_count()
    if threads > 61:  # to run in processors with 30+ cores
        threads = 61
    print('Running with ' + str(threads) + ' threads')
    p = multiprocessing.Pool(processes=threads - 1)

    return p
