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