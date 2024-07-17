import numpy as np
import matplotlib.pyplot as plt
import random
from util.util_funcs import azimuth_angle_clockwise
from numba import jit  # some functions uses numba to improve performance (some functions are not used anymore)

# This file is used for functions to calculated propagation models and calculations that are directly linked it
# All assist functions are structured to work with i x n x m matrix with i is the number of centers coordinates and -
# - n x m are the dimensions of the grid matrix to be calculated
# All other functions are structured with a j x i x n x m matrix format. With j been the number of different -
# antennas in one center, i is the number of center coordinates and n x m are the matrix dimensions
# Alternatively, the functions can calculate sample points if his positions are available in the samples variable -
# or just arrange the functions in a format that the vectorized functions will work


def generate_azimuth_map(lines, columns, centroids, samples=None, plot=False):
    # returns a centroids x lines x columns matrix with per pixel azimuth information from a centroid

    if samples is None:
        coord_map = np.indices((lines, columns))

        az_map = np.ndarray(shape=(centroids.shape[0], lines, columns))
        for i, centroid in enumerate(centroids):
            az_map[i] = azimuth_angle_clockwise(np.asarray([centroid[0], centroid[1]]),
                                                np.asarray([coord_map[0], coord_map[1]]))
    else:
        az_map = np.ndarray(shape=(centroids.shape[0], samples.shape[0]))
        for i, centroid in enumerate(centroids):
            az_map[i] = azimuth_angle_clockwise(np.asarray([centroid[0], centroid[1]]),
                                                       np.asarray([samples[:, 0], samples[:, 1]]))

    if plot:
        title = 'Azimuth map'
        plot_func(az_map, 1, centroids.shape[0], title)

    return az_map


def generate_euclidian_distance(lines, columns, centers, samples=None, plot=False):
    if samples is not None:
        dist_mtx = np.ndarray(shape=(centers.shape[0], samples.shape[0]))

        a_b = np.ndarray(shape=(centers.shape[0], 2))

        for i, coord in enumerate(samples):
            a_b[:, 0] = centers[:, 0] - coord[0]
            a_b[:, 1] = centers[:, 1] - coord[1]

            dist_mtx[:, i] = np.linalg.norm(a_b, axis=1)  # euclidean distance using L2 norm

        dist_mtx = np.where(dist_mtx == 0, 1, dist_mtx)
    else:
        coord_map = np.indices((lines, columns)).transpose((1, 2, 0))  # coordinates map used to calculate the distance
        dist_mtx = np.ndarray(shape=(centers.shape[0], lines, columns))  # calculated euclidean distance from n_centers to each pixel

        a_b = np.ndarray(shape=(centers.shape[0], 2))
        for line in coord_map:
            for coord in line:
                a_b[:, 0] = centers[:, 0] - coord[0]
                a_b[:, 1] = centers[:, 1] - coord[1]

                dist_mtx[:, coord[0], coord[1]] = np.linalg.norm(a_b, axis=1)  # euclidean distance using L2 norm

        dist_mtx = np.where(dist_mtx == 0, 1, dist_mtx)

    if plot:
        for i in range(centers.shape[0]):
            plt.matshow(dist_mtx[i])
            plt.title('Distance matrix: ' + str(i))
            plt.show()

    return dist_mtx


def generate_elevation_map(htx, hrx, d_euclid, cell_size, samples=None, plot=False):
    # returns a centroids x lines x columns matrix with per pixel elevation information from a centroid
    if samples is None:
        if d_euclid is None:
            pass
        theta = np.rad2deg(np.arctan((htx - hrx) / (d_euclid * cell_size)))
    else:
        d_euclid_s = d_euclid[:, samples[:, 0], samples[:, 1]]
        theta = np.rad2deg(np.rad2deg(np.arctan((htx - hrx) / (d_euclid_s * cell_size))))

    if plot:
        title = 'Elevation map'
        plot_func(theta, 1,  theta.shape[0], title)

    return theta


def generate_distance_map(eucli_dist_map, cell_size, htx, hrx, plot=False):
    # returns a centroids x lines x columns matrix with per pixel distance information from a centroid with htx height -
    # - to a coordinate with hrx height

    # converting the euclidean distance to actual distance between Tx and Rx
    dist_map = np.sqrt((eucli_dist_map * cell_size) ** 2 + (htx - hrx) ** 2)

    if plot:
        title = 'distance map'
        plot_func(dist_map, 1,  dist_map.shape[0], title)

    return np.where(dist_map == 0, 1, dist_map)

def generate_bf_gain(elevation_map, azimuth_map, base_station_list=None, sector_index = None):

    # this function works only with azimth and elevation in sample format
    if hasattr(base_station_list[0].antenna, 'beamforming_id'):  # checking if the antenna is a beamforming one
        if base_station_list is not None:
            gain_map = []
            for i, base_station in enumerate(base_station_list):
                sectors_gain = []
                if sector_index is not None:
                    sectors = [base_station.beam_sector_pattern[sector_index]]  # for some sectors
                else:
                    sectors = base_station.beam_sector_pattern  # for all sectors
                for sector, sector_antenna in enumerate(sectors):
                    if sector_index is not None:  # VER SE ISTO AQUI ESTÁ FUNCIONANDO!!!!
                        sector = [sector_index][sector]
                    gain_samples_sector = np.ndarray(shape=(sector_antenna.beams, elevation_map.shape[1]))
                    for beam in range(sector_antenna.beams):
                        for coordinate, _ in enumerate(elevation_map[i]):
                            gain_samples_sector[beam][coordinate] = sector_antenna.calculate_gain(beam, np.rint(azimuth_map[i][coordinate]
                                                            - base_station.sectors_pointing[sector]).astype(int), np.rint(180+elevation_map[i][coordinate]).astype(int))

                    sectors_gain.append(gain_samples_sector)

                gain_map.append(sectors_gain)

            return gain_map

        else:
            print('empty base_station_list !!!')

def generate_gain_map(antenna, elevation_map, azimuth_map, sectors_hor_pattern=None, sectors_ver_pattern=None, base_station_list=None):
    if base_station_list is not None:
        hor_gain = np.ndarray(shape=np.append(base_station_list[0].sectors_hor_pattern.shape[0], azimuth_map.shape))
        ver_gain = np.ndarray(shape=np.append(base_station_list[0].sectors_ver_pattern.shape[0], azimuth_map.shape))
        for i, base_station in enumerate(base_station_list):
            for j, hor_pattern in enumerate(base_station.sectors_hor_pattern):
                hor_gain[j, i] = np.interp(np.deg2rad(azimuth_map[i]), antenna.sigma, hor_pattern)
            for j, ver_pattern in enumerate(base_station.sectors_ver_pattern):
                ver_gain[j, i] = np.interp(np.deg2rad(elevation_map[i]), antenna.theta, ver_pattern)

    elif sectors_hor_pattern is not None and sectors_ver_pattern is not None:
        hor_gain = np.ndarray(shape=np.append(sectors_hor_pattern.shape[0], azimuth_map.shape))
        ver_gain = np.ndarray(shape=np.append(sectors_ver_pattern.shape[0], elevation_map.shape))

        # horizontal gain
        for i, hor_pattern in enumerate(sectors_hor_pattern):
            hor_gain[i] = np.interp(np.deg2rad(azimuth_map), antenna.sigma, hor_pattern)

        # vertical gain
        for i, ver_pattern in enumerate(sectors_ver_pattern):
            ver_gain[i] = np.interp(np.deg2rad(elevation_map), antenna.theta, ver_pattern)

    else:
        # horizontal gain
        hor_gain = np.interp(np.deg2rad(azimuth_map), antenna.sigma, antenna.hoz_pattern)
        # vertical gain
        ver_gain = np.interp(np.deg2rad(elevation_map), antenna.theta, antenna.ver_pattern)

    # total gain (dB)
    hor_gain = np.where(hor_gain == 0, 0.000001, hor_gain)
    ver_gain = np.where(ver_gain == 0, 0.000001, ver_gain)

    gain_map = antenna.gain + 10*np.log10(hor_gain*ver_gain)

    return gain_map

def generate_path_loss_map(eucli_dist_map, cell_size, prop_model, frequency, htx, hrx, samples=None, plot=False, **kwargs):

    # converting the euclidean distance to actual distance between Tx and Rx and using the actual distance for a cell size
    if samples is not None:
        eucli_dist_map = eucli_dist_map[:, samples[:, 0], samples[:, 1]]

    dist_map = generate_distance_map(eucli_dist_map, cell_size, htx, hrx)

    # calculating the prop model for each centroid for each cell of the grid
    if prop_model == 'free space':
        if 'var' in kwargs:
            var = kwargs['var']
            path_loss_map = fs_path_loss(dist_map/1000, frequency, var=var)
        else:
            path_loss_map = fs_path_loss(dist_map / 1000, frequency)
    elif prop_model == '3GPP UMA':
            path_loss_map = generate_uma_path_loss(eucli_dist_map,dist_map,hrx,htx,frequency)
    elif prop_model == '3GPP UMA O2I':
            path_loss_map = generate_uma_path_loss_o2i(eucli_dist_map,dist_map,hrx,htx,frequency)
    elif prop_model == 'WINNER2 C2':
            path_loss_map = generate_win2_path_loss_c2(eucli_dist_map,dist_map,hrx,htx,frequency)
    elif prop_model == 'WINNER2 C4':
            path_loss_map = generate_win2_path_loss_c4(eucli_dist_map,dist_map,hrx,htx,frequency)
    else:
        print('wrong path loss model !!! please see the available ones in: .....')

    if plot:
        n_centroids = eucli_dist_map.shape[0]
        title = 'path loss map using' + prop_model
        plot_func(path_loss_map, 1, n_centroids, title)

    return path_loss_map


def generate_rx_power_map(path_loss_map, azimuth_map, elevation_map, base_station, gain_map=None):

    if gain_map is None:  # can use custom gain map with custom gain configuration (from base_station object)
        gain_map = generate_gain_map(base_station.antenna, elevation_map, azimuth_map, base_station.sectors_hor_pattern,
                                     base_station.sectors_ver_pattern)

    rx_power_map = np.ndarray(shape=gain_map.shape)

    for i in range(base_station.n_sectors):
        rx_power_map[i] = base_station.tx_power + gain_map[i] - path_loss_map  # tx power in dBW, gain and path loss in dB

    return rx_power_map


def generate_snr_map(base_station, rx_power_map, samples=None, threshold=None, unified=False, plot=False):
    # snr_map = np.ndarray(shape=(rx_power_map.shape[0], rx_power_map.shape[2], rx_power_map.shape[3]))
    # noise_interf = np.ndarray(shape=rx_power_map[:, 0].shape)
    # coord_map = np.indices((rx_power_map.shape[2], rx_power_map.shape[3])).transpose((1, 2, 0))  # coordinates map used to calculate the distance
    k = 1.380649E-23  # Boltzmann's constante (J/K)
    t = 290  # absolute temperature
    noise_power = k * t * base_station.bw * 10E6  # thermal noise power calculation

    pw_map_lin = 10**(rx_power_map/10)
    pwr_sum = np.sum(pw_map_lin, axis=1, dtype=np.float64)
    max_pw = np.amax(pw_map_lin,  axis=1)
    noise_interf = pwr_sum - max_pw + noise_power
    snr_map = 10*np.log10(max_pw/noise_interf)

    # map = np.empty(shape=(rx_power_map[0][0].shape[0], rx_power_map[0][0].shape[1]))  # empty map that will store the calculations
    # for i in range(rx_power_map.shape[0]):
    #     max_pw = np.amax(10 ** (rx_power_map[i] / 10), axis=0)
    #     noise_interf = np.sum(10**(rx_power_map[i]/10), axis=0, dtype=np.float64) - max_pw + noise_power
    #     # noise_interf[i] = total_power - max_pw + noise_power
    #     snr_map[i] = 10*np.log10(max_pw/noise_interf)

    if (unified is not True) and (unified is not False):
        print('\'unified\' variable must be True or False !!!')
        return

    snr_map_unified = np.max(snr_map, axis=0)

    perf_grid = np.mean(snr_map_unified)

    if threshold is not None:
        perf_grid_criteria = np.sum(np.where(snr_map_unified >= threshold, 1, 0)) / (
                snr_map.shape[0] * snr_map.shape[1])

    if samples is not None:
        perf_samples = np.mean(snr_map_unified[samples[:, 0], samples[:, 1]])
        if threshold is not None:
            perf_samples_criteria = np.sum(np.where(snr_map_unified[samples[:, 0], samples[:, 1]] >= threshold, 1, 0))\
                                / samples.shape[0]

        if unified:
            if threshold is not None:
                return snr_map, snr_map_unified, perf_samples_criteria, perf_samples, perf_grid_criteria, perf_grid
            else:
                return snr_map, snr_map_unified, perf_samples, perf_grid
        else:
            if threshold is not None:
                return snr_map, perf_samples_criteria, perf_samples, perf_grid_criteria, perf_grid
            else:
                return snr_map, perf_samples, perf_grid
    else:
        if unified:
            if threshold is not None:
                return snr_map, snr_map_unified, perf_grid_criteria, perf_grid
            else:
                return snr_map, snr_map_unified, perf_grid
        else:
            if threshold is not None:
                return snr_map, perf_grid_criteria, perf_grid
            else:
                return snr_map, perf_grid



        # max_pw = np.unravel_index(np.argmax(rx_power_map[i], axis=0), rx_power_map[i][0].shape)
        # for line in coord_map:
        #     for coord in line:
        #         noise_interf[i][coord[0], coord[1]] = np.sum(10**(rx_power_map[i][np.arange(len(rx_power_map[i])) != max_pw[1][coord[0], coord[1]]]/10)) + noise_power
        #         map[coord[0], coord[1]] = rx_power_map[i][max_pw[1][coord[0], coord[1]]][coord[0], coord[1]] - 10*np.log10(noise_interf[i][coord[0], coord[1]])
        # snr_map[i] = map


    # if plot:
    #     title = 'SNR map'
    #     plot_func(snr_map, centroids, title)

    return snr_map


def generate_capcity_map(snr_map, bw, samples=None, threshold=None, unified=False, plot=False):
    bw = bw * 10E6
    capacity_map = bw * np.log2(1 + 10**(snr_map/10))  # simple shanon's law equation

    if (unified is not True) and (unified is not False):
        print('\'unified\' variable must be True or False !!!')
        return

    capacity_map_unified = np.max(capacity_map, axis=0)

    perf_grid = np.mean(capacity_map_unified)

    if threshold is not None:
        perf_grid_criteria = np.sum(np.where(capacity_map_unified >= threshold, 1, 0)) / (
                snr_map.shape[0] * snr_map.shape[1])

    if samples is not None:
        perf_samples = np.mean(capacity_map_unified[samples[:, 0], samples[:, 1]])
        print(np.round(np.std(capacity_map_unified[samples[:, 0], samples[:, 1]])/10E6,2))
        perf_samples_criteria = np.sum(np.where(capacity_map_unified[samples[:, 0], samples[:, 1]] >= threshold, 1, 0))\
                                / samples.shape[0]

        if unified:
            if threshold is not None:
                return capacity_map, capacity_map_unified, perf_samples_criteria, perf_samples, perf_grid_criteria, perf_grid
            else:
                return capacity_map, capacity_map_unified, perf_samples, perf_grid
        else:
            if threshold is not None:
                return capacity_map, perf_samples_criteria, perf_samples, perf_grid_criteria, perf_grid
            else:
                return capacity_map, perf_samples, perf_grid
    else:
        if unified:
            if threshold is not None:
                return capacity_map, capacity_map_unified, perf_grid_criteria, perf_grid
            else:
                return capacity_map, capacity_map_unified, perf_grid
        else:
            if threshold is not None:
                return capacity_map, perf_grid_criteria, perf_grid
            else:
                return capacity_map, perf_grid



    # if plot:
    #     title = 'capacity map'
    #     plot_func(capacity_map, len(capacity_map), title)


def plot_func(map = None, sectors = 1, centroids=1, title=''):
    # this is just a generic function to plot the different functions results
    for j in range(sectors):
        for i in range(centroids):
            if sectors == 1:
                plt.matshow(map[i],  origin='lower')
                plt.title(title + ' (' + str(i) + ')')
                plt.colorbar()
                plt.show()
            else:
                plt.matshow(map[j][i], origin='lower')
                plt.title(title + ' (' + str(j) + ' ' + str(i) + ')')
                plt.colorbar()
                plt.show()


# @jit(nopython=True, parallel=True)
# def fs_path_loss(d, f):  # simple free space path loss function
#     fspl = 40 * np.log10(d) + 20 * np.log10(f) + 92.45  # f in GHz and d in km
#
#     return fspl

# TODO - REFAZER COM AS VARIAVEIS ALFA BETA GAMA E VAR COMO ARGUMENTO !!!!!
def fs_path_loss(d, f, var=6):  # simple free space path loss function with lognormal component
    if var:
        log_n = np.random.lognormal(mean=0, sigma=np.sqrt(var), size=d.shape)  # lognormal variance from the mediam path loss
    else:
        log_n = 0

    pl = 40 * np.log10(d) + 20 * np.log10(f) + 92.45 + log_n  # f in GHz and d in km

    return pl

#################################### 3GPP CENÁRIOS UMA e 3GPP UMA O2I ###################################################

def calculate_los_prob_uma(d2d, hut):
    """
    Calcula a probabilidade de um percurso ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 338.901 v 17.1.0.
    
    d2d - Distância no eixo horizontal entre a BS e a UE (m) / no cenário outdoor-outdoor apenas!
    hut - Altura do UE (m)
    """
    # Ensure the height constraint
    if hut > 23:
        raise Exception("Altura do UE deve ser menor que 23 m")

    d2d = np.asarray(d2d)  # Ensure d2d is a NumPy array

    # Vectorized computation
    problos = np.zeros_like(d2d, dtype=float)

    # For d2d <= 18
    problos[d2d <= 18] = 1

    # For d2d > 18
    above_18 = d2d > 18
    if hut <= 13:
        c = 0
    else:
        c = np.power((hut - 13) / 10, 1.5)

    problos[above_18] = ((18 / d2d[above_18]) + np.exp(-d2d[above_18] / 63) * (1 - 18 / d2d[above_18])) * (
                    1 + c * 5 / 4 * np.power(d2d[above_18] / 100, 3) * np.exp(-d2d[above_18] / 150))

    return problos

#Create the UMA 3GPP Path Loss based on TR 38.901 O SHADOW FADING TÁ ZUADO/ DÁ ERRO DE EMPTY SLICE
def generate_uma_path_loss(d2d, d3d, hut, hbs, fc):
    """
    Calcula um percurso, considerado ele ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 38.901 v 17.1.0.\n
    fc - frequência em GHz \n
    hut -  Altura da UE (m) \n
    hbs - Altura da BS (m) \n
    d2d e d3d são as distâncias em (m)!
    """

    rng = np.random.default_rng(seed=42)
    c = 3*10**8 # Velocidade da luz (m/s)
    dbp = 4*(hbs-1)*(hut-1)*fc*10**9/c
    # Ploss = np.empty_like(d2d) # Cria array de vazio para preencher com os PLOS

    # o código abaixo só funciona se d2d e d3d foram ndarray
    problos = calculate_los_prob_uma(d2d, hut)
    PLOSS = np.zeros_like(d2d)

    # eu acredito que para essa linha não seja possível fazer vetorizado
    prop = np.array([[rng.choice(['LOS', 'NLOS'], p=[p, 1 - p]) for p in bs] for bs in problos.T]).T
    # prop = random.choices(["LOS", "NLOS"], weights=[problos, 1 - problos], k=d2d.shape[0])  # Simula se a propagação será LOS ou NLOS

    # criando uma matrix booleana que indica quando d2d<dbp
    less_dbp = d2d < dbp

    PLOSS[less_dbp] = 28 + 22 * np.log10(d3d[less_dbp]) + 20 * np.log10(fc)+shadow_fading(d2d[less_dbp],0,4) #PL1

    # agora quando d2d >= dbp (usando np.invert())
    PLOSS[~less_dbp]= 28 + 40 * np.log10(d3d[~less_dbp]) + 20 * np.log10(fc) \
                       -9*np.log10(np.power(dbp, 2)+np.power(hbs-hut,2))+shadow_fading(d2d[~less_dbp],0,4) #PL2

    # verifico quando prop não é PLOS, irá atualizar o  código utilizando o máximo entre o Pl1/PL2 ou PNLOS para todos os elementos do array que essa condição for TRUE
    not_los = prop != "LOS"

    PLOSS[not_los] = np.maximum(PLOSS[not_los],
                              13.54 + 39.08 * np.log10(d2d[not_los]) +
                              20 * np.log10(fc) - 0.6 * (hut - 1.5))+shadow_fading(d2d[not_los],0,6)
    
    return PLOSS 
 
def generate_uma_path_loss_o2i(d2d, d3d, hut, hbs, fc):
    """
    Calcula um percurso, considerado ele ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 38.901 v 17.1.0.\n
    fc - frequência em GHz \n
    hut -  Altura da UE (m) \n
    hbs - Altura da BS (m) \n
    d2d e d3d são as distâncias em (m)!
    AQUI, A MAIORIA DAS VARIÁVEIS TEM NOMES CONFUSOS OU NÃO ESTÃO ESPECIFICADAS! VOCÊ PREVISA REVER OS NOMES DE VARIÁVEIS PARA QUE
    ELES FAÇAM REFERÊNCIA A RECOMENDAÇÃO QUE ESTÁ UTILIZANDO!!!!
    """
    rng = np.random.default_rng(seed=42)
    c = 3*10**8 # Velocidade da luz (m/s)
    dbp = 4*(hbs-1)*(hut-1)*fc*10**9/c
    # Ploss = np.empty_like(d2d) # Cria array de vazio para preencher com os PLOS

    # o código abaixo só funciona se d2d e d3d foram ndarray
    problos = calculate_los_prob_uma(d2d, hut)
    pen_loss = o2i_pen_loss(d2d,fc)
    PLOSS = np.zeros_like(d2d)

    #o2i = np.array([rng.choice(["High", "Low"], p=[0.2, 0.8], size=dim.shape[1]) for _ in dim]).reshape(dim.shape)
    # eu acredito que para essa linha não seja possível fazer vetorizado
    prop = np.array([[rng.choice(['LOS', 'NLOS'], p=[p, 1 - p]) for p in bs] for bs in problos.T]).T
    # prop = random.choices(["LOS", "NLOS"], weights=[problos, 1 - problos], k=d2d.shape[0])  # Simula se a propagação será LOS ou NLOS

    # criando uma matrix booleana que indica quando d2d<dbp
    less_dbp = d2d < dbp

    PLOSS[less_dbp] = 28 + 22 * np.log10(d3d[less_dbp]) + 20 * np.log10(fc)+shadow_fading(d2d[less_dbp],0,4) #PL1

    # agora quando d2d >= dbp (usando np.invert())
    PLOSS[~less_dbp]= 28 + 40 * np.log10(d3d[~less_dbp]) + 20 * np.log10(fc) \
                       -9*np.log10(np.power(dbp, 2)+np.power(hbs-hut,2))+shadow_fading(d2d[~less_dbp],0,4) #PL2

    # verifico quando prop não é PLOS, irá atualizar o  código utilizando o máximo entre o Pl1/PL2 ou PNLOS para todos os elementos do array que essa condição for TRUE
    not_los = prop != "LOS"

    PLOSS[not_los] = np.maximum(PLOSS[not_los],
                              13.54 + 39.08 * np.log10(d2d[not_los]) +
                              20 * np.log10(fc) - 0.6 * (hut - 1.5))+shadow_fading(d2d[not_los],0,6)
    PLOSS += pen_loss
    return PLOSS

######################## FADING e O2I LOSSES ###################################

def shadow_fading(dim,m,std):
    """
    Função para calcular o efeito de perda por sombreamento em, baseada numa distribuição lognormal.Sendo:
    m: Média
    std: Desvio padrão
    """
    sf = np.random.lognormal(mean=m, sigma=std,size = dim.shape)
    sf = np.nan_to_num(sf, nan=0)
    return sf

def indoor_loss(dim):
    """Gera uma penetração O2I baseado no item 7.4.3.1 da TR 38.901. V.17.1.0"""
    d2in = np.random.uniform(low=0,high=25,size=dim.shape) #Distância indoor. Deve ser entre 0-25 m
    #if None:
    return 0.5*d2in
def high_loss_building_o2i_pen_loss(dim,f):
    """Gera uma penetração O2I baseado no item 7.4.3.1 da TR 38.901. V.17.1.0. A frequência f em GHz."""
    return 5-10*np.log10(0.7*np.power(10,-(23+0.3*f)/10)+0.3*np.power(10,-(5+4*f)/10))+ np.random.uniform(low=0,high=25,size=dim.shape)+np.random.normal(0,6.5,size=dim.shape)

def low_loss_building_o2i_pen_loss(dim,f):
    """Gera uma penetração O2I baseado no item 7.4.3.1 da TR 38.901. V.17.1.0. A frequência f em GHz."""
    return 5-10*np.log10(0.3*np.power(10,-(2+0.2*f)/10)+0.7*np.power(10,-(5+4*f)/10))+ np.random.uniform(low=0,high=25,size=dim.shape)+np.random.normal(0,4.4,size=dim.shape)

#Não usado
#def car_o2i_pen_loss(dim):
    """Gera uma penetração O2I baseado no item 7.4.3.2 da TR 38.901. V.17.1.0"""
    #return np.random.normal(9,5,size=dim.shape)

def o2i_pen_loss(dim,f):
    rng = np.random.default_rng(seed=42)

    o2i = np.array([rng.choice(["High", "Low"], p=[0.2, 0.8], size=dim.shape[1]) for _ in dim]).reshape(dim.shape)

    
    pen_loss = np.zeros_like(dim)

    is_high = o2i == "High"

    pen_loss[is_high] = (5-10*np.log10(0.7*np.power(10,-(23+0.3*f)/10)+0.3*np.power(10,-(5+4*f)/10))+ 
    np.random.uniform(low=0,high=25,size=dim[is_high].shape)+np.random.normal(0,6.5,size=dim[is_high].shape))

    #~ é equivalente a np.invert

    pen_loss[~is_high] = (5-10*np.log10(0.3*np.power(10,-(2+0.2*f)/10)+0.7*np.power(10,-(5+4*f)/10))+ 
    np.random.uniform(low=0,high=25,size=dim[~is_high].shape)+np.random.normal(0,4.4,size=dim[~is_high].shape))
    return pen_loss  

########################## WINNER 2 CENÁRIOS C2 E C4 ############################################

#@jit()
def calculate_los_prob_win2(d2d):
    """
    Calcula a probabilidade de um percurso ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado no projeto winner 2.\n
    d2d - Distância no eixo horizontal entre a BS e a UE (m) / no cenário outdoor-outdoor apenas!\n
    """
    problos = np.minimum(18/d2d,1)*(1-np.exp(-d2d/63))+np.exp(-d2d/63)

    return problos

#@jit() O SHADOW FADING TÁ ZUADO
def generate_win2_path_loss_c2(d2d, d3d, hut, hbs,fc):
    """
    Calcula um percurso, considerado ele ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 38.901 v 17.1.0.\n
    fc - frequência em GHz \n
    hut -  Altura da UE (m) \n
    hbs - Altura da BS (m) \n
    d2d e d3d são as distâncias em (m)!
    """
    rng = np.random.default_rng(seed=42)
    c = 3*10**8 # Velocidade da luz (m/s)
    dbp = 4*(hbs-1)*(hut-1)*fc*10**9/c
    PLOSS = np.zeros_like(d2d) # Cria array de vazio para preencher com os PLOS

    problos = calculate_los_prob_win2(d2d)
    
    #prop = np.array([rng.choice(['LOS', 'NLOS'], p=[p, 1 - p],size = problos.shape[1]) for _ in problos]).reshape(problos.shape)

    prop = np.array([[rng.choice(['LOS', 'NLOS'], p=[p, 1 - p]) for p in bs] for bs in problos.T]).T

    less_dbp = d2d < dbp

    # criando uma matrix booleana que indica quando d2d<dbp

    PLOSS[less_dbp] = 39+26*np.log10(d3d[less_dbp])+20*np.log10(fc/5.0)+shadow_fading(d2d[less_dbp],0,4) #PL1

    # agora quando d2d >= dbp (usando np.invert())
    PLOSS[~less_dbp] = 13.47+40*np.log10(d3d[~less_dbp])+6*np.log10(fc/5.0) \
                       -14.0*np.log10(hbs-1)- 14.0*np.log10(hut-1)+shadow_fading(d2d[~less_dbp],0,6)  #PL2

    # verifico quando prop não é PLOS
    not_los = prop != "LOS"

    PLOSS[not_los] = np.maximum(PLOSS[not_los],
                              13.54 + 39.08 * np.log10(d2d[not_los]) +
                              20 * np.log10(fc) - 0.6 * (hut - 1.5))+shadow_fading(d2d[not_los],0,6)

    return PLOSS
   
def generate_win2_path_loss_c4(d2d, d3d, hut, hbs, fc):
    """
    Calcula um percurso, considerado ele ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 38.901 v 17.1.0.\n
    fc - frequência em GHz \n
    hut -  Altura da UE (m) \n
    hbs - Altura da BS (m) \n
    d2d e d3d são as distâncias em (m)!
    """
    rng = np.random.default_rng(seed=42)
    din = np.random.uniform(0,25,size=d2d.shape)
    d = d2d+din #array
    nrfi = np.random.uniform(1,5,size=d2d.shape)
    hms = 3*nrfi+hut #array
    c = 3*10**8 # Velocidade da luz (m/s)
    dbp = 4*(hbs-1)*(hms-1)*fc*10**9/c
    PLOSS = np.zeros_like(d2d) # Cria array de vazio para preencher com os PLOS

    problos = calculate_los_prob_win2(d2d)

    prop = np.array([[rng.choice(['LOS', 'NLOS'], p=[p, 1 - p]) for p in bs] for bs in problos.T]).T

    less_dbp = d2d < dbp

    # criando uma matrix booleana que indica quando d2d<dbp

    PLOSS[less_dbp] = 39+26*np.log10(d3d[less_dbp])+20*np.log10(fc/5.0)+shadow_fading(d2d[less_dbp],0,4) #PL1

    # agora quando d2d >= dbp (usando np.invert())
    PLOSS[~less_dbp] = 13.47+40*np.log10(d3d[~less_dbp])+6*np.log10(fc/5.0) \
                        -14.0*np.log10(hbs-1)- 14.0*np.log10(hut-1)+shadow_fading(d2d[~less_dbp],0,6)  #PL2

    # verifico quando prop não é PLOS
    not_los = prop != "LOS"

    PLOSS[not_los] = np.maximum(PLOSS[not_los],
                              (44.9-6.55*np.log10(hbs))*np.log10(d3d[not_los])+31.46 \
                         + 5.83*np.log10(hbs)+23*np.log10(fc/5.0)+shadow_fading(d2d[not_los],0,8))

    PLOSS += 17.4 + 0.5*din -0.8*hms

    return PLOSS





######################
@jit()
def calc_snr(coord_map, noise_power, rx_power_map, max_pw, map, noise_interf):
    for line in coord_map:
        for coord in line:
            noise_interf[coord[0], coord[1]] = np.sum(10 ** (rx_power_map[
                                                                    np.arange(len(rx_power_map)) != max_pw[1][
                                                                        coord[0], coord[1]]] / 10)) + noise_power
            map[coord[0], coord[1]] = rx_power_map[max_pw[1][coord[0], coord[1]]][
                                          coord[0], coord[1]] - 10 * np.log10(noise_interf[coord[0], coord[1]])

    return map

