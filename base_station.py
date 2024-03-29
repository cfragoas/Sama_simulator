import numpy as np
import matplotlib.pyplot as plt
import copy, warnings
from models.scheduler.scheduler import Scheduler
from models.scheduler.master_scheduler import Master_scheduler
from numba import jit  # some functions uses numba to improve performance (some functions are not used anymore)


# BaseStation Class represents the relationship between the station resources (antenna and transmission characteristics)
# and the space and users around it

class BaseStation:
    def __init__(self, frequency, tx_power, tx_height, bw, n_sectors, antenna, gain, downtilts, plot=False):  # simple function, but will include sectors and MIMO in the future
        self.frequency = frequency
        self.tx_power = tx_power  # tx power in dBW
        self.tx_height = tx_height
        self.bw = bw/n_sectors
        self.n_sectors = n_sectors
        self.sector_bw = self.bw / self.n_sectors

        if np.size(downtilts) == 1:  # if the downtilts variable is not array type
            downtilts = np.zeros(shape=n_sectors) + downtilts  # setting all the array values the same
        self.downtilts = downtilts  # this variable must have the shape of the same size of n_sectors

        self.antenna = antenna  # antenna object
        self.scheduler = None  # scheduler object
        self.tdd_mux = Master_scheduler()  # time/frequency multiplexer


        # initializing calculated variables
        self.sectors_pointing = None


        if not hasattr(self.antenna, 'beamforming_id'):
            self.sectors_hor_pattern = []  # rotated antenna patterns for each of the base station sectors
            self.sectors_ver_pattern = []  # tilted elevation pattern

            self.generate_ant_pattern()
            self.generate_sector_pattern(plot)
        else:
            self.beam_sector_pattern = []
            self.dwn_active_beams = None
            self.up_active_beams = None
            self.active_beams_index = None
            if hasattr(self.antenna, 'beams'):
                self.beams = self.antenna.beams
            else:
                self.beams = None

    def initialize_mux(self, simulation_time=None, up_tdd_time=None):
        self.tdd_mux.create_tdd_scheduler(simulation_time=simulation_time, up_tdd_time=up_tdd_time)

    def initialize_dwn_up_scheduler(self, downlink_specs=None, uplink_specs=None):
        # this function will instantiate the schedulers, downlink and uplink separately. It will instantiate using the
        # configuration found in the downlink_specs and uplink_specs dictionaries create from the param.yml file
        if downlink_specs is not None or uplink_specs is not None:
            if self.tdd_mux.dwn_tdd_time != 0:
                if downlink_specs is not None:
                    self.tdd_mux.create_downlink(scheduler_typ=downlink_specs['scheduler_typ'],
                                                 bs_index=downlink_specs['bs_index'], bw=downlink_specs['bw'],
                                                 time_slot=downlink_specs['time_slot'],
                                                 simulation_time=downlink_specs['simulation_time'],
                                                 t_min=downlink_specs['t_min'],
                                                 bw_slot=downlink_specs['bw_slot'],c_target=downlink_specs['criteria'],
                                                 tx_power=downlink_specs['tx_power'])
            if self.tdd_mux.up_tdd_time != 0:
                if uplink_specs is not None:
                    self.tdd_mux.create_uplink(scheduler_typ=uplink_specs['scheduler_typ'],
                                               bs_index=uplink_specs['bs_index'], bw=uplink_specs['bw'],
                                               time_slot=uplink_specs['time_slot'],
                                               simulation_time=uplink_specs['simulation_time'],
                                               t_min=uplink_specs['t_min'],
                                               bw_slot=uplink_specs['bw_slot'], c_target=uplink_specs['criteria'],
                                               tx_power=uplink_specs['tx_power'])
        else:
            raise ValueError('downlink or uplink configurations are not found, please verify the parameter file')

    def beam_configuration(self, az_map, elev_map=None): # change the beam configuration according to the grid if beamforing is used
        # always in sample list!!!
        if elev_map is None:  # in case of one dimension beamforming
            elev_map = np.zeros(shape=az_map.shape)

        self.sectors_phi_range = np.arange(360 / self.n_sectors, 360.1, 360 / self.n_sectors)
        self.sectors_pointing = np.arange(360 / (2*self.n_sectors), 360, 360 / self.n_sectors)
        lower_bound = 0

        for sector, higher_bound in enumerate(self.sectors_phi_range):
            range_sector = np.where((az_map > lower_bound) & (az_map <= higher_bound))
            self.antenna.change_beam_configuration(point_phi=np.rint(az_map[range_sector]-self.sectors_pointing[sector])
                                                   , point_theta=-np.rint(elev_map[range_sector]))
            self.beam_sector_pattern.append(copy.deepcopy(self.antenna))

            # self.antenna.change_beam_configuration(point_phi=az_map[range_sector], point_theta=elev_map[range_sector])
            # self.beams = self.antenna.beams  # NOT USING - FOR FUTURE CALCULATIONS
            lower_bound = higher_bound

    def add_active_beam(self, sector, beams, n_users, uplink, downlink):
        # this function turns the beams in active position and indicates in a matrix (downlink and uplink are separated)
        if hasattr(self, 'dwn_active_beams'):
            if downlink:
                if self.dwn_active_beams is None:
                    self.dwn_active_beams = np.zeros(shape=(self.antenna.beams, self.n_sectors))
        if hasattr(self, 'up_active_beams'):
            if uplink:
                if self.up_active_beams is None:
                    self.up_active_beams = np.zeros(shape=(self.antenna.beams, self.n_sectors))
        else:
            print('Active beam list not found! Is the antenna object a beamforming one?')
            return
        for beam_index, beam in enumerate(beams):
            if downlink:
                self.dwn_active_beams[beam][sector] = n_users[beam_index]
            elif uplink:
                self.up_active_beams[beam][sector] = n_users[beam_index]

    def clear_active_beams(self, downlink=False, uplink=False):
        # This function will erases the active beam matrices (uplink and downlink)
        if downlink:
            self.dwn_active_beams = None
        elif uplink:
            self.up_active_beams = None

    def sector_beam_pointing_configuration(self, n_beams):
        # sectors_pointing = np.arange(360/(2*self.n_sectors), 360.1, 360/self.n_sectors)
        self.beams_pointing = np.array([])
        sector_apperture = 360/self.n_sectors
        beams_pointing_0 = np.arange(sector_apperture/(2*n_beams), sector_apperture+0.1, sector_apperture/n_beams)
        self.beams_pointing = beams_pointing_0
        for i in range(1, self.n_sectors):
            self.beams_pointing = np.append(self.beams_pointing, beams_pointing_0 + sector_apperture*i)

    def generate_ant_pattern(self):  # used for sector antennas WITHOUT beamforming
        # horizontal_beamwidth = 360/self.n_sectors
        self.antenna.hor_beamwidth = 360/self.n_sectors
        # self.antenna = self.antenna.ITU1336(gain=gain, frequency=self.frequency, hor_beamwidth=horizontal_beamwidth, ver_beamwidth=10)
        self.antenna.build_diagram()

    def generate_sector_pattern(self, plot=False):  # used for pivot the antennas WITHOUT beamforming
        self.sectors_pointing = np.arange(360 / (2*self.n_sectors), 360, 360 / self.n_sectors)
        for sector_pointing in self.sectors_pointing:
            rotated_azim = self.antenna.sigma + np.deg2rad(sector_pointing)
            rotated_azim = np.where(rotated_azim < 0, rotated_azim + (2 * np.pi), rotated_azim)
            rotated_azim = np.where(rotated_azim > 2 * np.pi, rotated_azim - (2 * np.pi), rotated_azim)
            sector_pattern = np.interp(rotated_azim, self.antenna.sigma, self.antenna.hoz_pattern)
            self.sectors_hor_pattern.append(sector_pattern)

        for downtilt in self.downtilts:
            tilted_pattern = self.antenna.theta - np.deg2rad(downtilt)
            tilted_pattern = np.where(tilted_pattern < 0, tilted_pattern + (2*np.pi), tilted_pattern)
            tilted_pattern = np.where(tilted_pattern > 2 * np.pi, tilted_pattern - (2*np.pi), tilted_pattern)
            tilted_elev_pattern = np.interp(tilted_pattern, self.antenna.theta, self.antenna.ver_pattern)
            self.sectors_ver_pattern.append(tilted_elev_pattern)

        self.sectors_hor_pattern = np.asarray(self.sectors_hor_pattern)
        self.sectors_ver_pattern = np.asarray(self.sectors_ver_pattern)

        if plot:
            plt.polar(self.antenna.sigma, self.sectors_hor_pattern[0])
            for i in range(self.n_sectors - 1):
                plt.plot(self.antenna.sigma, self.sectors_hor_pattern[i+1])
            plt.title('Base station sectors horizontal patterns')
            # plt.show(block=False)

            plt.figure()
            plt.polar(self.antenna.theta, self.sectors_ver_pattern[0])
            for i in range(self.n_sectors - 1):
                plt.plot(self.antenna.theta, self.sectors_ver_pattern[i+1])
            plt.title('Base station sectors vertical patterns')
            plt.show()

    def change_downtilt(self, downtilts, plot=False):
        if np.size(downtilts) == 1:  # if the downtilts variable is not array type
            downtilts = np.zeros(shape=self.n_sectors) + downtilts  # setting all the array values the same
        self.downtilts = downtilts  # this variable must have the shape of the same size of n_sectors
        self.sectors_hor_pattern = []  # rotated antenna patterns for each of the base station sectors
        self.sectors_ver_pattern = []  # tilted elevation pattern
        self.generate_sector_pattern(plot)


