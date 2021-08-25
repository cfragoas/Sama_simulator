import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import cm



class Element_ITU1336:
    # original from recomendation ITU F1336-5 item 3.1.1

    def __init__(self, max_gain, theta_3db, phi_3db, freq, plot=False):
        # alfa = np.arctan(np.tan(theta)/np.sin(phi))
        # psi_alfa = 1/np.sqrt((np.cos(alfa)/phi_3dB)**2 + (np.sen(alfa)/theta_3dB))
        # psi = np.arccos(np.cos(phi)*np.cos(theta))
        # x = psi/psi_alfa
        self.max_gain = max_gain
        self.theta_3db = theta_3db
        self.phi_3db = phi_3db
        self.freq = freq

        self.phi = np.arange(0, 180)
        self.phi = np.where(self.phi == 0, 0.1, self.phi)
        self.theta = np.arange(-90, 270)
        self.theta = np.where(self.theta == 0, 0.1, self.theta)

        self.kp = 0.7
        self.kh = 0.8
        self.kv = 0.7

        self.g_ref_h = self.generate_horizontal_pattern(self.phi)
        self.g_ref_v = self.generate_vertical_pattern(self.theta)
        self.r = self.calc_compression_ratio(self.g_ref_h)

        self.g_ref_h = np.append(self.g_ref_h, np.flip(self.g_ref_h))
        self.phi = np.arange(0, 360)

        # self.g_ref_v = np.append(self.g_ref_v[90:360], self.g_ref_v[0:90])
        self.theta = np.arange(0, 360)

        self.r = np.append(self.r, np.flip(self.r))

        self.gain_pattern = np.ndarray(shape=(self.phi.shape[0], self.theta.shape[0]))
        self.calc_gain_pattern()

        if plot:
            self.plot()

    def generate_horizontal_pattern(self, phi):
        # horizontal pattern
        g_ref_h = np.ndarray(shape=phi.shape)
        x_h = abs(phi) / self.phi_3db

        lambda_kh = 3 * (1 - 0.5 ** (-self.kh))

        # for frequencies between 1 and 6 GHz:
        if 0.4 <= self.freq < 70:
            g180 = -12 + 10 * np.log10(1 + 9 * self.kp) - 15 * np.log10(180 / self.theta_3db)

            i0 = np.where(x_h <= 0.5)
            g_ref_h[i0] = -12 * x_h[i0] ** 2
            i1 = np.where(x_h > 0.5)
            g_ref_h[i1] = -12 * x_h[i1] ** (2 - self.kh) - lambda_kh

            g_ref_h = np.maximum(g_ref_h, g180)

        else:
            print('frequency should be between 400 Mhz and 70 GHz!!!')

        return g_ref_h

    def generate_vertical_pattern(self, theta):
        # vertical pattern
        g_ref_v = np.ndarray(shape=theta.shape)
        x_k = np.sqrt(1 - 0.36 * self.kv)
        x_v = abs(theta) / self.theta_3db

        c = (10 * np.log10((((180 / self.theta_3db) ** 1.5) * (4 ** (-1.5) + self.kv)) / (1 + 8 * self.kp))) \
            / (np.log10(22.5 / self.theta_3db))  # attenuation incline factor
        lambda_kv = 12 - c * np.log10(4) - 10 * np.log10((4 ** (-1.5)) + self.kv)

        # for frequencies between 1 and 6 GHz:
        if 0.4 <= self.freq < 70:
            g180 = -12 + 10 * np.log10(1 + 9 * self.kp) - 15 * np.log10(180 / self.theta_3db)

            i0 = np.where(x_v < x_k)
            g_ref_v[i0] = -12 * x_v[i0] ** 2
            i1 = np.where((x_k <= x_v) & (x_v < 4))
            g_ref_v[i1] = -12 + 10 * np.log10((x_v[i1] ** -1.5) + self.kv)
            i2 = np.where((x_v >= 4) & (x_v <= 90 / self.theta_3db))
            g_ref_v[i2] = -lambda_kv - c * np.log10(x_v[i2])
            i3 = np.where(x_v >= 90 / self.theta_3db)
            g_ref_v[i3] = g180


        else:
            print('frequency should be between 400 Mhz and 70 GHz!!!')

        return g_ref_v

    def calc_compression_ratio(self, g_ref_h):
        # horizontal gain compression ratio R
        g0 = 0
        g180 = -12 + 10 * np.log10(1 + 9 * self.kp) - 15 * np.log10(180 / self.theta_3db)

        # g_hr_180 = self.generate_horizontal_pattern(np.asarray([1, 180 / self.phi_3db]))[1]  # ghr(180/theta_3db) from eq 2a2
        r = (g_ref_h - g180) / (g0 - g180)

        return r

    def calc_gain_pattern(self):
        for i, _ in enumerate(self.phi):
            for j, _ in enumerate(self.theta):
                self.gain_pattern[i, j] = self.max_gain + self.g_ref_h[i] + self.r[i]*self.g_ref_v[j]  # pattern for all thetas and phis

        # plt.matshow(self.gain_pattern)
        # plt.show()

    def plot(self):
        ax = plt.subplot(111, projection='polar')
        ax.plot(np.deg2rad(self.phi), self.g_ref_h)
        # ax.set_rticks([0, -3, -6, -10, -20, -30, -60])
        ax.plot(np.deg2rad(self.theta), self.g_ref_v)
        ax.plot(np.deg2rad(self.phi), self.r)

        plt.show()

class Element_ITU2101:
    def __init__(self, max_gain, phi_3db, theta_3db, front_back_h, sla_v, plot=False):
        self.max_gain = max_gain # maximum gain of the element
        self.phi_3db = phi_3db
        self.theta_3db = theta_3db
        self.front_back_h = front_back_h  # element front back ratio
        self.sla_v = sla_v  # element vertical side lobe attenuation

        self.g_ref_h = None
        self.g_ref_v = None

        self.theta_list = np.arange(-90, 270)
        self.phi_list = np.arange(0, 180)

        self._generate_horizontal_pattern()
        self._generate_vertical_pattern()

        self.g_ref_h = np.append(self.g_ref_h, np.flip(self.g_ref_h))
        self.phi_list = np.arange(0, 360)

        # self.g_ref_v = np.append(self.g_ref_v[90: 360], self.g_ref_v[0: 90])
        # self.theta_list = np.arange(0, 360)

        self.gain_pattern = np.ndarray(shape=(self.phi_list.shape[0], self.theta_list.shape[0]))

        self._generate_gain_pattern()


        if plot:
            self.plot()

    def _generate_horizontal_pattern(self):
        self.g_ref_h = -np.minimum(12*(self.phi_list/self.phi_3db)**2, self.front_back_h)

    def _generate_vertical_pattern(self):
        self.g_ref_v = -np.minimum(12*((self.theta_list-90)/self.theta_3db)**2, self.sla_v)

    def _generate_gain_pattern(self):

        for i, phi in enumerate(self.phi_list):
            for j, theta in enumerate(self.theta_list):
                self.gain_pattern[i, j] = self.max_gain - np.minimum(-1*(self.g_ref_h[i] + self.g_ref_v[j]), self.front_back_h)

        # plt.matshow(self.gain_pattern)
        # plt.show()
        pass

    def plot(self):
        ax = plt.subplot(111, projection='polar')
        ax.plot(np.deg2rad(self.phi_list), self.g_ref_h)
        # ax.set_rticks([0, -3, -6, -10, -20, -30, -60])
        ax.plot(np.deg2rad(self.theta_list), self.g_ref_v)

        plt.show()

class beamforming_antenna():
    def __init__(self, ant_element, frequency, n_rows, n_columns, horizontal_spacing, vertical_spacing, point_theta=None, point_phi=None):
        self.ant_element = ant_element
        self.frequency = frequency
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        # c = 299792458  # speed of light
        # wavelgt = c/self.frequency
        self.dh = self.horizontal_spacing
        self.dv = self.vertical_spacing

        self.point_theta = point_theta
        self.point_theta = point_phi

        self.phi = np.arange(0, 360)
        self.theta = np.arange(0, 360)
        self.v_vec = None
        self.w_vec = None
        self._w_vec = np.ndarray(shape=(np.array(point_theta).shape[0], self.n_rows, self.n_columns),  dtype=complex)
        self.beam_gain = np.ndarray(shape=(np.array(point_theta).shape[0], self.phi.shape[0], self.theta.shape[0]))

        # for beam, [phi_tilt, theta_tilt] in enumerate(zip(point_phi, point_theta)):  # precalculating the weight vector
        #     self._w_vec[beam] = self._weight_vector(phi_tilt, theta_tilt)


        self.calculate_pattern(point_phi, point_theta)

    def _superposition_vector(self, phi, theta):
        rows = np.arange(self.n_rows) + 1
        columns = np.arange(self.n_columns) + 1
        theta = theta + 90
        # phi = phi - 180
        self.v_vec = np.exp(1j * 2 * np.pi * ((rows[:, np.newaxis] - 1) * self.dv * np.cos(np.deg2rad(theta)) +
                             (columns - 1) * self.dh * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))))

    def _weight_vector(self, point_phi, point_theta):
        rows = np.arange(self.n_rows) + 1
        columns = np.arange(self.n_columns) + 1
        # point_theta = -point_theta
        point_phi = -point_phi
        w_vec = (1 / np.sqrt(self.n_rows * self.n_columns)) * \
                     np.exp(1j * 2 * np.pi * ((rows[:, np.newaxis] - 1) * self.dv * np.sin(np.deg2rad(point_theta))
                            - (columns - 1) * self.dh * np.cos(np.deg2rad(point_theta)) * np.sin(np.deg2rad(point_phi))))
        return w_vec

    def calculate_gain(self, beam, phi, theta):  # NOT USED FOR NOW
        self._superposition_vector(phi, theta)
        gain = self.ant_element.gain_pattern[phi, theta] + 10*np.log10(abs(np.sum(self.w_vec * self.v_vec))**2)
        return gain

    def calculate_pattern(self, point_phi, point_theta, plot=False):
        # theta_list = np.arange(0, 360)
        # phi_list = np.arange(0, 360)

        self.point_theta = point_theta
        self.point_phi = point_phi

        for beam, [phi_tilt, theta_tilt] in enumerate(zip(point_phi, point_theta)):
            self._weight_vector(phi_tilt, theta_tilt)
            print(self.w_vec.shape)
            for phi in self.phi:
                for theta in self.theta:
                    self._superposition_vector(phi, theta)
                    self.beam_gain[beam, phi, theta] = self.ant_element.gain_pattern[phi, theta] + 10*np.log10(abs(np.sum(self.w_vec * self.v_vec))**2)

                # x =np.append(self.beam_gain[beam, phi, 180:180+theta_tilt], self.beam_gain[beam, phi, theta_tilt:180])
                # self.beam_gain[beam,phi,:] = np.append(x, np.flip(x))

            if plot:
                self.plot()
                # plt.plot(self.phi, self.beam_gain[beam, :, 180-theta_tilt])
                # plt.ylim(bottom=-30)
                # plt.grid(linestyle='--')
                # plt.title('phi')
                # plt.show()
                #
                # # plt.polar(np.deg2rad(self.theta), 10**(self.beam_gain[beam, phi_tilt,:]/10))
                # plt.plot(self.theta-180, self.beam_gain[beam, phi_tilt,:])
                # plt.ylim(bottom=-30)
                # plt.grid(linestyle='--')
                # plt.title('theta')
                # plt.show()

                # if the user wantts to plot a 2d gain map
                # plt.matshow(self.beam_gain[beam])
                # plt.show()

    def plot(self):
        if self.beam_gain is None:
            self.calculate_pattern(self.point_phi, self.point_theta)
        else:
            for beam, [phi_tilt, theta_tilt] in enumerate(zip(self.point_phi, self.point_theta)):
                plt.plot(self.phi, self.beam_gain[beam, :, 180 - theta_tilt])
                plt.ylim(bottom=-30)
                plt.grid(linestyle='--')
                plt.title('phi')
                plt.show()

                # plt.polar(np.deg2rad(self.theta), 10**(self.beam_gain[beam, phi_tilt,:]/10))
                plt.plot(self.theta - 180, self.beam_gain[beam, phi_tilt, :])
                plt.ylim(bottom=-30)
                plt.grid(linestyle='--')
                plt.title('theta')
                plt.show()
