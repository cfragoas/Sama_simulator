import datetime, pickle, os, logging
import  matplotlib.pyplot as plt

from antennas.beamforming import Beamforming_Antenna
from antennas.ITU2101_Element import Element_ITU2101
from base_station import BaseStation
from user_eq import User_eq
from make_grid import Grid
from macel2 import Macel

def load_data(name_file):
    folder = os.path.dirname(__file__)
    folder = '\\'.join(folder.split('\\')[:-1])
    folder += '\\output\\'
    folder += name_file

    with open(folder, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()

    return(data_dict)

def save_data(path = None, data_dict = None):
    if not path:
        folder = os.path.dirname(__file__)
        folder = '\\'.join(folder.split('\\')[:-1])
        folder += '\\output\\'
        date = datetime.datetime.now()
        name_file = date.strftime('%x') + '-' + date.strftime('%X') + '.pkl'
        name_file = name_file.replace('/', '_').replace(':', '_')
        path = folder + name_file

        return path

    else:
        if data_dict and type(data_dict) is dict:
            with open(path, 'wb') as f:
                pickle.dump([data_dict], f)
                f.close()
                logging.info('Saved/updated file ' + path)
        else:
            logging.error('data_dictionary not provided!!!!')

def plot(mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw, std_user_bw,
         individual = False, save = False):
    if individual:
        if save:
            pass

    else:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)
        fig.suptitle('Metrics evolution by BS number - ' + str(max_iter) + ' iterations')
        ax1.plot(mean_snr)
        ax1.set_title('Mean SNIR')
        ax2.plot(std_snr)
        ax2.set_title('std SNIR')
        ax3.plot(mean_cap)
        ax3.set_title('Mean Capacity (Mbps)')
        ax4.plot(std_cap)
        ax4.set_title('std Capacity (Mbps)')
        ax5.plot(mean_user_time)
        ax5.set_title('Mean user time (s)')
        ax6.plot(std_user_time)
        ax6.set_title('std user time (s)')
        ax7.plot(mean_user_bw)
        ax7.set_title('Mean user bw (MHz)')
        ax8.plot(std_user_bw)
        ax8.set_title('std user bw (MHz)')
        fig.tight_layout()
        plt.show()
        if save:
            pass


def teste (n_bs, grid, macel):
    grid.make_points(dist_type='gaussian', samples=50, n_centers=4, random_centers=False,
                          plot=False)  # distributing points around centers in the grid
    ue = User_eq(positions=grid.grid, height=1.5)  # creating the user equipament object
    macel.set_ue(ue=ue)
    snr_cap_stats = macel.place_and_configure_bs(n_centers=n_bs)

    return(snr_cap_stats)




if __name__ == '__main__':
    n_bs = 5
    samples = 50



    grid = Grid()  # grid object
    grid.make_grid(1000, 1000)

    element = Element_ITU2101(max_gain=5, phi_3db=65, theta_3db=65, front_back_h=30, sla_v=30, plot=False)
    beam_ant = Beamforming_Antenna(ant_element=element, frequency=10, n_rows=8, n_columns=8, horizontal_spacing=0.5,
                                   vertical_spacing=0.5)
    base_station = BaseStation(frequency=3.5, tx_power=50, tx_height=30, bw=300, n_sectors=3, antenna=beam_ant, gain=10,
                     downtilts=0, plot=False)
    base_station.sector_beam_pointing_configuration(n_beams=10)
    macel = Macel(grid=grid, prop_model='free space', criteria=0, cell_size=30, base_station=base_station)

    x = teste(n_bs, grid, macel)
    print(x)
