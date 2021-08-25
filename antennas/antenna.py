import numpy as np
import matplotlib.pyplot as plt


class Antenna:  # mother class with general variables and functions

    def __init__(self, gain):
        self.gain = gain

        #  calculated variables
        self.hoz_pattern = None
        self.ver_pattern = None
        self.sigma = None
        self.theta = None


    def plot(self, az_plot='polar', elev_plot='rect'):
        # plot function plots the horizontal and vertical diagrams in polar or rectangular format
        if az_plot == elev_plot:
            if az_plot == 'polar':
                plt.polar(self.sigma, self.hoz_pattern)
                plt.plot(self.theta, self.ver_pattern)
                plt.title('Horizontal + vertical pattern')
                plt.show()
            elif az_plot == 'rect':
                plt.plot(self.sigma, self.hoz_pattern)
                plt.plot(self.sigma, self.ver_pattern)
                plt.title('Horizontal + vertical pattern')
                plt.show()
            else:
                print('please use \'polar\' or \'rect\' for the az_plot and elve_plot variables')

        else:
            if az_plot == 'polar' and elev_plot == 'rect':
                plt.figure(1)
                plt.polar(self.sigma, self.hoz_pattern)
                plt.title('Horizontal pattern')
                plt.show(block=False)
                plt.figure(2)
                plt.plot(self.theta, self.ver_pattern)
                plt.title('Vertical pattern')
                plt.show(block=False)
            elif az_plot == 'rect' and elev_plot == 'polar':
                plt.figure(1)
                plt.plot(self.sigma, self.hoz_pattern)
                plt.title('Horizontal pattern')
                plt.show(block=False)
                plt.figure(2)
                plt.polar(self.theta, self.ver_pattern)
                plt.title('Vertical pattern')
                plt.show(block=False)
            else:
                print('please use \'polar\' or \'rect\' for the az_plot and elve_plot variables')


class ITU1336(Antenna):
    def __init__(self, gain, frequency, hor_beamwidth, ver_beamwidth, build=False):
        super().__init__(gain)
        self.frequency = frequency
        self.hor_beamwidth = hor_beamwidth
        self.ver_beamwidth = ver_beamwidth

        if build:
            self.build_diagram()

    def build_diagram(self, plot=True):
        # reference pattern - itu 1336-3

        # radiation pattern in horizontal plane
        sigma_3db = np.deg2rad(self.hor_beamwidth)
        self.sigma = np.arange(np.deg2rad(-180), np.deg2rad(179), np.deg2rad(0.1))
        b = - np.log(0.5) * ((2 / sigma_3db) ** 2)
        self.hoz_pattern = np.e ** (-b * (self.sigma ** 2))

        # radiation pattern in vertical plane
        theta_3db = np.deg2rad(self.ver_beamwidth)
        self.theta = np.arange(np.deg2rad(-180), np.deg2rad(179), np.deg2rad(0.1))
        a = - np.log(0.5) * ((2 / theta_3db) ** 2)
        self.ver_pattern = np.e ** (-a * (self.theta ** 2))

        # normalizing and sorting sigma and theta between 0 and 360 (helps in other functions)
        self.sigma[self.sigma < 0] = self.sigma[self.sigma < 0] + 2 * np.pi
        self.theta[self.theta < 0] = self.theta[self.theta < 0] + 2 * np.pi
        indices_sigma = np.argsort(self.sigma)
        indices_theta = np.argsort(self.theta)
        self.sigma = self.sigma[indices_sigma]
        self.hoz_pattern = self.hoz_pattern[indices_sigma]
        self.theta = self.theta[indices_theta]
        self.ver_pattern = self.ver_pattern[indices_theta]

        if plot:
            self.plot()

