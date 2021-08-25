import numpy as np
import matplotlib.pyplot as plt

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