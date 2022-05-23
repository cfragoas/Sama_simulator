import numpy as np
import matplotlib.pyplot as plt

class Beamforming_Antenna():
    def __init__(self, ant_element, frequency, n_rows, n_columns, horizontal_spacing, vertical_spacing, point_theta=None, point_phi=None):
        self.ant_element = ant_element
        self.frequency = frequency  # not used
        self.n_rows = n_rows
        self.n_columns = n_columns
        # c = 299792458  # speed of light
        # wavelgt = c/self.frequency
        self.dh = horizontal_spacing
        self.dv = vertical_spacing
        self.beamforming_id = True # just a identifier of a beamforming antenna

        self.point_theta = point_theta
        self.point_phi = point_phi

        self.phi = np.arange(0, 360)
        self.theta = np.arange(0, 360)
        self.v_vec = None
        self.w_vec = None

        if point_theta is not None and point_phi is not None:
            self.beams = len(point_theta)
            self.w_vec = np.ndarray(shape=(np.array(point_theta).shape[0], self.n_rows, self.n_columns), dtype=complex)
            self.beam_gain = np.ndarray(shape=(np.array(point_theta).shape[0], self.phi.shape[0], self.theta.shape[0]))

            for beam, [phi_tilt, theta_tilt] in enumerate(zip(point_phi, point_theta)):  # precalculating the weight vector
                self.w_vec[beam] = self._weight_vector(phi_tilt, theta_tilt)

    def change_beam_configuration(self, point_theta, point_phi):
        self.point_theta = point_theta
        self.point_phi = point_phi
        self.beams = len(point_theta)

        self.w_vec = np.ndarray(shape=(np.array(point_theta).shape[0], self.n_rows, self.n_columns), dtype=complex)
        for beam, [phi_tilt, theta_tilt] in enumerate(zip(point_phi, point_theta)):  # precalculating the weight vector
            self.w_vec[beam] = self._weight_vector(phi_tilt, theta_tilt)


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

    def calculate_gain(self, beam, phi, theta):
        if self.w_vec is None:
            if self.point_phi is not None and self.point_theta is not None:
                self.w_vec = np.ndarray(shape=(np.array(self.point_theta).shape[0], self.n_rows, self.n_columns), dtype=complex)
                for beam, [phi_tilt, theta_tilt] in enumerate(zip(self.point_phi, self.point_theta)):  # calculating the weight vector
                    self.w_vec[beam] = self._weight_vector(phi_tilt, theta_tilt)
            else:
                print('need to define beam theta and phi first!!!')

        self._superposition_vector(phi, theta)
        gain = self.ant_element.gain_pattern[phi, theta] + 10*np.log10(abs(np.sum(self.w_vec[beam] * self.v_vec))**2)

        return gain

    def calculate_pattern(self, point_phi=None, point_theta=None, plot=False):

        if point_phi is not None and point_theta is not None:  # if one wants to change the beams
            self.point_theta = point_theta
            self.point_phi = point_phi
            self.w_vec = None

        if self.w_vec is None:
            self.w_vec = np.ndarray(shape=(np.array(self.point_theta).shape[0], self.n_rows, self.n_columns), dtype=complex)
            for beam, [phi_tilt, theta_tilt] in enumerate(zip(self.point_phi, self.point_theta)):  # calculating the weight vector
                self.w_vec[beam] = self._weight_vector(phi_tilt, theta_tilt)

        for beam, _ in enumerate(self.point_phi):
            for phi in self.phi:
                for theta in self.theta:
                    self.beam_gain[beam, phi, theta] = self.calculate_gain(beam=beam, phi=phi, theta=theta)
                    # self._superposition_vector(phi, theta)
                    # self.beam_gain[beam, phi, theta] = self.ant_element.gain_pattern[phi, theta] + \
                    #                                    10*np.log10(abs(np.sum(self.w_vec[beam] * self.v_vec))**2)

        if plot:
            self.plot()

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