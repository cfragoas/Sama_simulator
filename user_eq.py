import numpy as np

class User_eq:
    def __init__(self, positions, height):
        self.positions = positions
        self.height = height
        self.bs_candidates = None
        self.gain_matrix = None

        self.sector_map = None

        # calculated variables
        self.ue_bs = None  # bs|beam

    def acquire_bs_and_beam(self, ch_gain_map, sector_map):
        self.sector_map = sector_map.astype(int)
        self.ue_bs = np.ndarray(shape=(ch_gain_map.shape[1], 3))  # bs|beam|sector

        for ue_index in range(ch_gain_map.shape[1]):
            # self.ue_bs[ue_index] = np.unravel_index(np.argmax(ch_gain_map[:, ue_index]), ch_gain_map[:, ue_index].shape)
            self.ue_bs[ue_index] = np.concatenate((np.array(
                np.unravel_index(np.argmax(ch_gain_map[:, ue_index]), ch_gain_map[:, ue_index].shape)), np.array(
                self.sector_map[np.unravel_index(np.argmax(ch_gain_map[:, ue_index]), ch_gain_map[:, ue_index].shape)[
                                    0], ue_index])), axis=None)

            self.ue_bs = self.ue_bs.astype(int)