import numpy as np

class Metrics:
    def __init__(self):
        self.up_simulation_time = None
        self.up_time_slot = None
        self.dwn_simulation_time = None
        self.dwn_time_slot = None
        # UE metrics
        # uplink
        self.up_cap = None
        self.up_snr = None
        self.up_user_time = None
        self.up_user_bw = None
        self.up_meet_criteria = None

        # downlink
        self.dwn_cap = None
        self.dwn_snr = None
        self.dwn_user_time = None
        self.dwn_user_bw = None
        # self.dwn_meet_criteria = None

        # BS metrics
        # uplink
        self.up_act_beams_nmb = None
        self.up_user_per_bs = None

        # downlink
        self.dwn_act_beams_nmb = None
        self.dwn_user_per_bs = None

        # criteria - if needed
        self.up_criteria = None
        self.dwn_criteria = None
        self.up_satisfied_ue = None
        self.up_cnt_satisfied_ue = None
        self.dwn_satisfied_ue = None
        self.dwn_cnt_satisfied_ue = None
        self.up_cap_deficit = None
        self.dwn_cnt_satisfied_ue = None
        self.dwn_cap_deficit = None

    def store_uplink_metrics(self, cap=None, snr=None, t_index=None, base_station_list=None, n_bs=None, n_ues=None,
                             simulation_time=None, time_slot=None, criteria=None):
        # this function stores the uplink metrics during the simulation time
        if (self.up_cap is None) or (self.up_snr is None) or (self.up_user_time is None) or (self.up_user_bw is None) \
                or (self.up_meet_criteria is None):
            if (simulation_time is not None) and (time_slot is not None) and (n_ues is not None) and (n_bs is not None):
                self.create_uplink_matrices(simulation_time=simulation_time, time_slot=time_slot,
                                            n_ues=n_ues, n_bs=n_bs, criteria=criteria)
            else:
                # error here
                pass
        else:
            self.up_cap[:, t_index] = cap
            self.up_snr[:, t_index] = snr
            self.up_user_time[:, t_index] = ~np.isnan(cap)
            # self.up_meet_criteria[t_index] = count_satisfied_ue
            bw = np.zeros(shape=self.up_cap.shape[0])
            for bs_index, base_station in enumerate(base_station_list):
                self.up_act_beams_nmb[bs_index, t_index] = np.mean(np.count_nonzero(base_station.dwn_active_beams, axis=0))
                self.up_user_per_bs[bs_index, t_index] = np.sum(base_station.dwn_active_beams)
                bw[base_station.tdd_mux.up_scheduler.freq_scheduler.user_bw != 0] = \
                    base_station.tdd_mux.up_scheduler.freq_scheduler.user_bw[base_station.tdd_mux.up_scheduler.freq_scheduler.user_bw != 0]

            self.up_user_bw[:, t_index][~np.isnan(cap)] = bw[~np.isnan(cap)]
            if self.up_criteria is not None:
                acc_ue_cap = np.nansum(self.up_cap, axis=1)  # accumulated capacity
                self.up_satisfied_ue = np.where(acc_ue_cap >= self.up_criteria)[0]  # UEs that satisfied the capacity goal
                self.up_cnt_satisfied_ue[t_index] = self.up_satisfied_ue.size
                self.up_cap_deficit = self.up_criteria - acc_ue_cap
                self.up_cap_deficit = np.where(self.up_cap_deficit < 0, 1E-6, self.up_cap_deficit)

    def store_downlink_metrics(self, cap=None, snr=None, t_index=None, base_station_list=None, n_bs=None, n_ues=None,
                             simulation_time=None, time_slot=None, criteria=None):
        # this function stores the downlink metrics during the simulation time
        if (self.dwn_cap is None) or (self.dwn_snr is None) or (self.dwn_user_time is None) or (self.dwn_user_bw is None) or (self.dwn_cnt_satisfied_ue is None):
            if (simulation_time is not None) and (time_slot is not None) and (n_ues is not None) and (n_bs is not None):
                self.create_downlink_matrices(simulation_time=simulation_time, time_slot=time_slot,
                                              n_ues=n_ues, n_bs=n_bs, criteria=criteria)
        else:
            # storing the UE metrics
            self.dwn_cap[:, t_index] = cap
            self.dwn_snr[:, t_index] = snr
            self.dwn_user_time[:, t_index] = ~np.isnan(cap)
            bw = np.zeros(shape=self.dwn_cap.shape[0])
            # storing the BS metrics
            for bs_index, base_station in enumerate(base_station_list):
                self.dwn_act_beams_nmb[bs_index, t_index] = np.mean(np.count_nonzero(base_station.dwn_active_beams, axis=0))
                self.dwn_user_per_bs[bs_index, t_index] = np.sum(base_station.dwn_active_beams)
                bw[base_station.tdd_mux.dwn_scheduler.freq_scheduler.user_bw != 0] = \
                    base_station.tdd_mux.dwn_scheduler.freq_scheduler.user_bw[base_station.tdd_mux.dwn_scheduler.freq_scheduler.user_bw != 0]

            self.dwn_user_bw[:, t_index][~np.isnan(cap)] = bw[~np.isnan(cap)]
            if self.dwn_criteria is not None:
                acc_ue_cap = np.nansum(self.dwn_cap, axis=1)  # accumulated capacity
                self.dwn_satisfied_ue = np.where(acc_ue_cap >= self.dwn_criteria)[0]  # UEs that satisfied the capacity goal
                self.dwn_cnt_satisfied_ue[t_index] = self.dwn_satisfied_ue.size
                self.dwn_cap_deficit = self.dwn_criteria - acc_ue_cap
                self.dwn_cap_deficit = np.where(self.dwn_cap_deficit < 0, 1E-6, self.dwn_cap_deficit)

    def create_uplink_matrices(self, simulation_time, time_slot, n_ues, n_bs, criteria):
        n_slots = np.ceil(simulation_time / time_slot).astype(int)
        self.up_simulation_time = simulation_time
        self.up_time_slot = time_slot
        # initialize UE metrics matrices
        self.up_cap = np.zeros(shape=[n_ues, n_slots])
        self.up_snr = np.zeros(shape=[n_ues, n_slots])
        self.up_user_time = np.zeros(shape=[n_ues, n_slots])
        self.up_user_bw = np.zeros(shape=[n_ues, n_slots])
        self.up_meet_criteria = np.zeros(n_slots)
        # setting all values to NaN to avoid value confusion
        self.up_cap.fill(np.nan)
        self.up_snr.fill(np.nan)
        self.up_user_time.fill(np.nan)
        self.up_user_bw.fill(np.nan)
        # self.up_meet_criteria.fill(np.nan)
        # initialize BS metrics matrices
        self.up_act_beams_nmb = np.zeros(shape=[n_bs, n_slots])
        self.up_user_per_bs = np.zeros(shape=[n_bs, n_slots])
        self.up_act_beams_nmb.fill(np.nan)
        self.up_user_per_bs.fill(np.nan)
        # initialize RAN metrics matrices
        self.up_cnt_satisfied_ue = np.zeros(shape=n_slots, dtype=int)
        if criteria is not None:
            self.up_criteria = criteria

    def create_downlink_matrices(self, simulation_time, time_slot, n_ues, n_bs, criteria):
        n_slots = np.ceil(simulation_time / time_slot).astype(int)
        self.dwn_simulation_time = simulation_time
        self.dwn_time_slot = time_slot
        # initialize UE metrics matrices
        self.dwn_cap = np.zeros(shape=[n_ues, n_slots])
        self.dwn_snr = np.zeros(shape=[n_ues, n_slots])
        self.dwn_user_time = np.zeros(shape=[n_ues, n_slots])
        self.dwn_user_bw = np.zeros(shape=[n_ues, n_slots])
        # self.dwn_meet_criteria = np.zeros(n_slots)
        # setting all values to NaN to avoid value confusion
        self.dwn_cap.fill(np.nan)
        self.dwn_snr.fill(np.nan)
        self.dwn_user_time.fill(np.nan)
        # self.dwn_user_bw.fill(np.nan)
        # self.dwn_meet_criteria.fill(np.nan)
        # initialize BS metrics matrices
        self.dwn_act_beams_nmb = np.zeros(shape=[n_bs, n_slots])
        self.dwn_user_per_bs = np.zeros(shape=[n_bs, n_slots])
        self.dwn_act_beams_nmb.fill(np.nan)
        self.dwn_user_per_bs.fill(np.nan)
        # initialize RAN metrics matrices
        self.dwn_cnt_satisfied_ue = np.zeros(shape=n_slots, dtype=int)
        if criteria is not None:
            self.dwn_criteria = criteria

    def create_uplink_metrics_dataframe(self, output_typ, active_ue, cluster_centroids, ue_pos, ue_bs_table, dist_map,
                                        scheduler_typ=None):

        if scheduler_typ == 'BCQI':  # todo ver aqui essa parada
            val_snr_line = np.nansum(self.up_snr, axis=1) != 0
            # todo definir mean snr
            mean_snr = 10 * np.log10(np.nanmean(self.up_snr[val_snr_line, :], axis=1))
        else:
            mean_snr = 10 * np.log10(np.nansum(self.up_snr[active_ue], axis=1))  # todo - WTF?????
        cap_sum = np.nansum(self.up_cap[active_ue], axis=1)  # ME ARRUMA !!!
        # cap_sum = np.nansum(cap[self.ue.active_ue], axis=1)/(self.base_station_list[0].beam_timing_sequence.shape[1])  # ME ARRUMA !!!
        mean_act_beams = np.mean(self.up_act_beams_nmb, axis=1)
        mean_user_bs = np.mean(self.up_user_per_bs, axis=1)
        user_time = np.nansum(self.up_user_time[active_ue], axis=1) * (self.up_time_slot/self.up_simulation_time)  # ME ARRUMA !!!
        # user_time = np.nansum(user_time[self.ue.active_ue], axis=1) / (self.base_station_list[0].beam_timing_sequence.shape[1])  # ME ARRUMA !!!
        positions = [np.round(cluster_centroids).astype(int)]  # todo ver o que fazer com o objecto cluster

        # simple stats data

        mean_mean_snr = np.mean(mean_snr)
        std_snr = np.std(mean_snr)
        # min_mean_snr = np.min(mean_mean_snr)
        # max_mean_snr = np.max(mean_mean_snr)
        mean_cap = np.mean(cap_sum)
        std_cap = np.std(cap_sum)
        # min_mean_cap = np.min(cap_sum)
        # max_mean_cap = np.max(cap_sum)
        mean_user_time = np.mean(user_time)
        std_user_time = np.std(user_time)
        # min_user_time = np.min(user_time)
        # max_user_time = np.max(user_time)
        mean_user_bw = np.nanmean(self.up_user_bw[active_ue])
        std_user_bw = np.nanstd(self.up_user_bw[active_ue])
        # min_user_bw = np.min(user_bw)
        # max_user_bw = np.max(user_bw)
        # FAZER AS OUTRAS MÉTRICAS !!!!
        # this part of the code is to check if one or multiple UEs have reached the criteria
        total_meet_criteria = None
        ran_cap_per_time = np.nansum(self.up_cap, axis=0)

        if self.up_criteria is not None:
            total_meet_criteria = np.sum(cap_sum >= self.up_criteria) / ue_bs_table.shape[0]
            deficit = self.up_criteria - cap_sum
            mean_deficit = np.mean(deficit)
            std_deficit = np.std(deficit)
            norm_deficit = 1 - cap_sum / self.up_criteria
            mean_norm_deficit = np.mean(norm_deficit)
            std_norm_deficit = np.mean(norm_deficit)

        if total_meet_criteria or total_meet_criteria == 0:
            # snr_cap_stats = [mean_mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw,
            #                  std_user_bw, total_meet_criteria, mean_deficit, std_deficit, mean_norm_deficit,
            #                  std_norm_deficit]
            snr_cap_stats = {'mean_snr':mean_mean_snr, 'std_snr':std_snr, 'mean_cap':mean_cap, 'std_cap':std_cap,
                             'mean_user_time':mean_user_time, 'std_user_time':std_user_time, 'mean_user_bw':mean_user_bw,
                             'std_user_bw':std_user_bw, 'total_meet_criteria':total_meet_criteria,
                             'mean_deficit':mean_deficit, 'std_deficit':std_deficit,
                             'mean_norm_deficit':mean_norm_deficit,
                             'std_norm_deficit':std_norm_deficit}
        else:
            # snr_cap_stats = [mean_mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw,
            #                  std_user_bw]
            snr_cap_stats = {'mean_mean_snr':mean_mean_snr, 'std_snr':std_snr, 'mean_cap':mean_cap, 'std_cap':std_cap,
                             'mean_user_time':mean_user_time, 'std_user_time':std_user_time, 'mean_user_bw':mean_user_bw,
                             'std_user_bw':std_user_bw}

        # preparing 'raw' data to export
        # ue_pos = self.cluster.features
        if total_meet_criteria or total_meet_criteria == 0:
            raw_data_dict = {'bs_position': positions, 'ue_position': ue_pos, 'ue_bs_table': ue_bs_table,
                             'snr': mean_snr, 'cap': cap_sum,
                             'user_bs': mean_user_bs, 'act_beams': mean_act_beams, 'user_time': user_time,
                             'user_bw': np.nanmean(self.up_user_bw[active_ue], axis=1), 'deficit': deficit,
                             'norm_deficit': norm_deficit, 'meet_criteria': self.up_meet_criteria,
                             'ran_cap_per_time': ran_cap_per_time, 'dist_map': dist_map}
        else:
            raw_data_dict = {'bs_position': positions, 'ue_position': ue_pos, 'snr': mean_snr, 'cap': cap_sum,
                             'user_bs': mean_user_bs, 'act_beams': mean_act_beams,
                             'user_time': user_time, 'user_bw': np.nanmean(self.up_user_bw, axis=1),
                             'ran_cap_per_time': ran_cap_per_time, 'dist_map': dist_map}

        if output_typ == 'simple':
            # return snr_cap_stats
            return {'snr_cap_stats': snr_cap_stats}
        if output_typ == 'complete':
            return {'snr_cap_stats': snr_cap_stats, 'raw_data_dict': raw_data_dict}
            # return snr_cap_stats, raw_data_dict
        if output_typ == 'raw':
            # return raw_data_dict
            return {'raw_data_dict': raw_data_dict}

    def create_downlink_metrics_dataframe(self, output_typ, active_ue, cluster_centroids, ue_pos, ue_bs_table, dist_map,
                                          scheduler_typ=None):
        if scheduler_typ == 'BCQI':  # todo ver aqui essa parada
            val_snr_line = np.nansum(self.dwn_snr, axis=1) != 0
            # todo definir mean snr
            # mean_snr = 10 * np.log10(np.nanmean(10 ** (self.dwn_snr[val_snr_line, :] / 10), axis=1))
            mean_snr = 10 * np.log10(np.nanmean(self.dwn_snr[val_snr_line, :], axis=1))
        else:
            # mean_snr = 10 * np.log10(np.nansum(10 ** (self.dwn_snr[active_ue] / 10), axis=1))  # todo - WTF?????
            mean_snr = 10 * np.log10(np.nanmean(self.dwn_snr[active_ue], axis=1))
        cap_sum = np.nansum(self.dwn_cap[active_ue], axis=1)  # ME ARRUMA !!!
        mean_act_beams = np.nanmean(self.dwn_act_beams_nmb, axis=1)
        mean_user_bs = np.nanmean(self.dwn_user_per_bs, axis=1)
        user_time = np.nansum(self.dwn_user_time[active_ue], axis=1) * (self.dwn_time_slot / self.dwn_simulation_time)  # ME ARRUMA !!!
        # user_time = np.nansum(user_time[self.ue.active_ue], axis=1) / (self.base_station_list[0].beam_timing_sequence.shape[1])  # ME ARRUMA !!!
        positions = [np.round(cluster_centroids).astype(int)]  # todo ver o que fazer com o objecto cluster

        # simple stats data
        mean_mean_snr = np.mean(mean_snr)
        std_snr = np.std(mean_snr)
        # min_mean_snr = np.min(mean_mean_snr)
        # max_mean_snr = np.max(mean_mean_snr)
        mean_cap = np.mean(cap_sum)
        std_cap = np.std(cap_sum)
        # min_mean_cap = np.min(cap_sum)
        # max_mean_cap = np.max(cap_sum)
        mean_user_time = np.mean(user_time)
        std_user_time = np.std(user_time)
        # min_user_time = np.min(user_time)
        # max_user_time = np.max(user_time)
        mean_user_bw = np.nanmean(self.dwn_user_bw[active_ue])
        std_user_bw = np.nanstd(self.dwn_user_bw[active_ue])
        # min_user_bw = np.min(user_bw)
        # max_user_bw = np.max(user_bw)
        # FAZER AS OUTRAS MÉTRICAS !!!!
        # this part of the code is to check if one or multiple UEs have reached the criteria
        total_meet_criteria = None
        ran_cap_per_time = np.nansum(self.dwn_cap, axis=0)

        if self.dwn_criteria is not None:
            total_meet_criteria = np.sum(cap_sum >= self.dwn_criteria) / ue_bs_table.shape[0]
            deficit = self.dwn_criteria - cap_sum
            mean_deficit = np.mean(deficit)
            std_deficit = np.std(deficit)
            norm_deficit = 1 - cap_sum / self.dwn_criteria
            mean_norm_deficit = np.mean(norm_deficit)
            std_norm_deficit = np.mean(norm_deficit)

        if total_meet_criteria or total_meet_criteria == 0:
            # snr_cap_stats = [mean_mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw,
            #                  std_user_bw, total_meet_criteria, mean_deficit, std_deficit, mean_norm_deficit,
            #                  std_norm_deficit]
            snr_cap_stats = {'mean_snr': mean_mean_snr, 'std_snr': std_snr, 'mean_cap': mean_cap, 'std_cap': std_cap,
                             'mean_user_time': mean_user_time, 'std_user_time': std_user_time,
                             'mean_user_bw': mean_user_bw,
                             'std_user_bw': std_user_bw, 'total_meet_criteria': total_meet_criteria,
                             'mean_deficit': mean_deficit, 'std_deficit': std_deficit,
                             'mean_norm_deficit': mean_norm_deficit,
                             'std_norm_deficit': std_norm_deficit}
        else:
            # snr_cap_stats = [mean_mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw,
            #                  std_user_bw]
            snr_cap_stats = {'mean_mean_snr': mean_mean_snr, 'std_snr': std_snr, 'mean_cap': mean_cap,
                             'std_cap': std_cap,
                             'mean_user_time': mean_user_time, 'std_user_time': std_user_time,
                             'mean_user_bw': mean_user_bw,
                             'std_user_bw': std_user_bw}

        # preparing 'raw' data to export
        # ue_pos = self.cluster.features
        if total_meet_criteria or total_meet_criteria == 0:
            raw_data_dict = {'bs_position': positions, 'ue_position': ue_pos, 'ue_bs_table': ue_bs_table,
                             'snr': mean_snr, 'cap': cap_sum,
                             'user_bs': mean_user_bs, 'act_beams': mean_act_beams, 'user_time': user_time,
                             'user_bw': np.nanmean(self.dwn_user_bw[active_ue], axis=1), 'deficit': deficit,
                             'norm_deficit': norm_deficit, 'meet_criteria': self.dwn_cnt_satisfied_ue,
                             'ran_cap_per_time': ran_cap_per_time, 'dist_map': dist_map}
        else:
            raw_data_dict = {'bs_position': positions, 'ue_position': ue_pos, 'snr': mean_snr, 'cap': cap_sum,
                             'user_bs': mean_user_bs, 'act_beams': mean_act_beams,
                             'user_time': user_time, 'user_bw': np.nanmean(self.dwn_user_bw, axis=1),
                             'ran_cap_per_time': ran_cap_per_time, 'dist_map': dist_map}

        if output_typ == 'simple':
            # return snr_cap_stats
            return {'snr_cap_stats': snr_cap_stats}
        if output_typ == 'complete':
            return {'snr_cap_stats': snr_cap_stats, 'raw_data_dict': raw_data_dict}
            # return snr_cap_stats, raw_data_dict
        if output_typ == 'raw':
            # return raw_data_dict
            return {'raw_data_dict': raw_data_dict}