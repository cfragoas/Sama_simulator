import numpy as np
from numba import jit


def highestPowerOf2(n):
    return (np.log2(n & (~(n - 1))))

@jit(nopython=True)
def azimuth_angle_clockwise(x1, x2):
    dx = x2[0] - x1[0]  # Difference in x coordinates
    dy = x2[1] - x1[1]  # Difference in y coordinates
    theta = np.rad2deg(np.arctan2(dy, dx))  # Angle between p1 and p2 in radians
    theta = np.where(theta > 0, theta, theta + 360)  # converting to positive degrees

    return theta

def elevation_angle(centroid, x, htx, hrx, cell_size, dist_matrix):
    d_euclid = dist_matrix[centroid][x[0], x[1]] * cell_size  # euclidean distance
    theta = np.rad2deg(np.arctan((htx-hrx)/d_euclid))

    return theta

def shannon_cap():
    return

def shannon_bw(bw, tx_power, channel_state, c_target):
    # this function will return the needed bw for a target capacity
    # bw in hearts, c_target in bits/s, tx_power in dBW, channel_state in dB
    k = 1.380649E-23  # Boltzmann's constant (J/K)
    t = 290  # absolute temperature
    pw_noise_bw = k * t * bw  # noise power
    # it is important here that tx_pw been in dBW (not dBm!!!)
    tx_pw = 10 ** (tx_power / 10)  # converting from dBW to watt
    snr = (tx_pw * 10 ** (channel_state/10)) / pw_noise_bw  # signal to noise ratio (linear, not dB) todo - checar se precisa verser 10^x/10 no channel state
    # bw_need = 2 ** (c_target[best_cqi_ue] / snr) - 1
    bw_need = c_target / (np.log2(1 + snr)) # needed bw to achieve the capacity target

    return bw_need
