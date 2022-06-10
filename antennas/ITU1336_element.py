import numpy as np
import matplotlib.pyplot as plt

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
