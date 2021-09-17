import numpy as np
import matplotlib.pyplot as plt
import copy

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

        # initializing calculated variables
        self.sectors_pointing = None
        if not hasattr(self.antenna, 'beamforming_id'):
            self.sectors_hor_pattern = []  # rotated antenna patterns for each of the base station sectors
            self.sectors_ver_pattern = []  # tilted elevation pattern

            self.generate_ant_pattern()
            self.generate_sector_pattern(plot)
        else:
            self.beam_sector_pattern = []
            self.active_beams = None
            self.active_beams_index = None
            self.beam_timing = None
            self.beam_timing_sequence = None
            self.beam_bw = None
            if hasattr(self.antenna, 'beams'):
                self.beams = self.antenna.beams
            else:
                self.beams = None

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

    def add_active_beam(self, sector, beams, n_users):
        if hasattr(self, 'active_beams'):
            if self.active_beams is None:
                self.active_beams = np.zeros(shape=(self.antenna.beams, self.n_sectors))
        else:
            print('Active beam list not found! Is the antenna object a beamforming one?')
            return
        for beam_index, beam in enumerate(beams):
            self.active_beams[beam][sector] = n_users[beam_index]

    def clean_active_beams(self):
        self.active_beams = None

    def generate_beam_timing(self, simulation_time, time_slot):
        # self.beam_timing = np.ndarray(shape=self.n_sectors)
        self.beam_timing = [None] * self.n_sectors
        for sector_index, sector in enumerate(self.beam_timing):
            sector = np.where(self.active_beams[:, sector_index] != 0)[0]
            np.random.shuffle(sector)  # randomizing the beam timing sequence
            self.beam_timing[sector_index] = sector  # I really dont know why this line is need to this code to work!!!

        self.beam_timing_sequence = np.zeros(shape=(self.n_sectors, np.round(simulation_time/time_slot).astype(int))) + self.antenna.beams
        for time in np.arange(0, simulation_time, time_slot):
            self.next_active_beam()
            for sector_index, sector in enumerate(self.beam_timing):
                if self.beam_timing[sector_index].size != 0:
                    self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][self.active_beams_index[sector_index].astype(int)]

        self.beam_timing_sequence = self.beam_timing_sequence.astype(int)

    def next_active_beam(self):
        if self.active_beams_index is None:
            self.active_beams_index = np.zeros(shape=self.n_sectors)
        else:
            self.active_beams_index += 1
            for sector_index, beam in enumerate(self.active_beams_index):
                if beam > len(self.beam_timing[sector_index])-1:
                    self.active_beams_index[sector_index] = 0

    def generate_beam_bw(self):
        import warnings
        warnings.filterwarnings("ignore")
        self.beam_bw = np.where(self.active_beams != 0, self.bw / self.active_beams, 0)
        warnings.simplefilter('always')

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


