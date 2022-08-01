import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import copy

from util.data_management import load_data, create_subfolder, extract_parameter_from_raw, group_ue, ue_relative_index
from make_grid import Grid
from matplotlib import cm

def plot_histograms(name_file, max_iter, global_parameters, n_bs=None):
    data_dict = load_data(name_file=name_file)
    if n_bs is None:
        n_bs = data_dict['BSs'][-1]  # picking the last simulation
    bs_data_index = np.where(np.array(data_dict['BSs']) == n_bs)[0][0]

    # checking if a downlink or uplink processing
    type = 'downlink'
    cap = np.concatenate([x['cap'] for x in data_dict['raw_data'][bs_data_index]])
    if type == 'downlink':
        criteria = global_parameters['downlink_scheduler']['criteria']
    if type == 'uplink':
        criteria = global_parameters['uplink_scheduler']['criteria']

    path = create_subfolder(name_file=name_file, n_bs=n_bs)

    beam_sec_groupings = group_ue(data_dict=data_dict, bs_data_index=bs_data_index)[0]

    rel_index_tables = ue_relative_index(data_dict=data_dict, bs_data_index=bs_data_index)[0]

    beam_cap = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='cap', bs_data_index=bs_data_index,
                                          concatenate=False)
    # beam capacity calculation
    acc_beam_cap = []
    # for index in range(max_iter):
    for index, groupings in enumerate(beam_sec_groupings['ue_per_beam']):
        for beam_ues in groupings:
            rel_beam_ues = np.array(rel_index_tables[index]['relative_index'])[
                np.isin(np.array(rel_index_tables[index]['ue_bs_index']), beam_ues)]
            acc_beam_cap.extend([beam_cap[index][rel_beam_ues].sum()])

    default_histogram(data=acc_beam_cap, n_bs=n_bs, path=path, title='Capacity per beam (Mbps)', save_name='beam_cap',
                      bins=100, binrange=(0, 100))

    # sector capacity calculation
    acc_sec_cap = []
    for index, groupings in enumerate(beam_sec_groupings['ue_per_sector']):
        for beam_ues in groupings:
            rel_beam_ues = np.array(rel_index_tables[index]['relative_index'])[
                np.isin(np.array(rel_index_tables[index]['ue_bs_index']), beam_ues)]
            acc_sec_cap.extend([beam_cap[index][rel_beam_ues].sum()])

    default_histogram(data=acc_sec_cap, n_bs=n_bs, path=path, title='Capacity per sector (Mbps)', save_name='sec_cap',
                      bins=100, binrange=(0, 1000))
    plt.close('all')

    # capacity x distance
    raw_dist = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='dist_map',
                                          bs_data_index=bs_data_index, concatenate=False)
    ue_bs_indexes = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='ue_bs_table',
                                               bs_data_index=bs_data_index, concatenate=False)
    ue_bs_indexes = [np.array(x['bs_index'][x['bs_index'] != -1]) for x in ue_bs_indexes]

    # raw_dist = np.concatenate([
    #     raw_dist[i][ue_bs_indexes[i]][np.array(x['ue_bs_index'])] for i, x in enumerate(rel_index_tables)])
    # raw_dist = np.concatenate([raw_dist[i][ue_bs_indexes[i], np.array(x['ue_bs_index'])] for i, x in enumerate(rel_index_tables)])
    raw_dist = np.concatenate([raw_dist[i][ue_bs_indexes[i], np.array(x['ue_bs_index'])] for i, x in enumerate(rel_index_tables)])
    raw_cap = np.concatenate(extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='cap',
                                         bs_data_index=bs_data_index,concatenate=False))

    title = 'Capacity x Distance for ' + str(n_bs) + ' BSs'
    max_dist = (np.sqrt(global_parameters['roi_param']['grid_columns'] * global_parameters['roi_param']['grid_lines']) *
                global_parameters['roi_param']['cel_size']) * 1.05/1000
    default_scatter(x=raw_dist/1000, y=raw_cap, path=path, title=title, n_bs=n_bs, dpi=100, subplot=None,
                    save_name='cap_x_dist', ylabel='Capacity (Mbps)', xlabel='Distance (km)', xlim=max_dist,
                    ylim=criteria*1.2)


    # default plots

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=100, figsize=(40, 30))
    fig.suptitle('Metrics using ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')

    # SINR plot
    snr = np.concatenate([x['snr'] for x in data_dict['raw_data'][bs_data_index]])
    ax1 = default_histogram(data=snr, n_bs=n_bs, subplot=ax1, path=path, title='SNIR (dB)', save_name='snr_',
                      bins=100, binrange=(-20, 20))

    # Capacity plot
    ax2 = default_histogram(data=cap, n_bs=n_bs, subplot=ax2, path=path, title='Throughput (Mbps)',
                      save_name='cap_', bins=100, binrange=(0, np.ceil(criteria * 1.05).astype(int)))

    # UE per BS plot
    user_bs = np.concatenate([x['user_bs'] for x in data_dict['raw_data'][bs_data_index]])
    n_ue = global_parameters['macel_param']['n_samples'] * global_parameters['macel_param']['n_centers']
    ax3 = default_histogram(data=user_bs, n_bs=n_bs, subplot=ax3, path=path, title='Number of UEs per BS',
                      save_name='user_bs_', bins=100, binrange=(0, n_ue))

    # Number of active beams plot
    act_beams = np.concatenate([x['act_beams'] for x in data_dict['raw_data'][bs_data_index]])
    n_beams = global_parameters['bs_param']['n_beams']
    ax4 = default_histogram(data=act_beams, n_bs=n_bs, subplot=ax4, path=path, title='Number of Active beams per BS',
                      save_name='act_beams_', bins=n_beams, binrange=(0, n_beams))

    # Active UE time plot
    user_time = np.concatenate([x['user_time'] for x in data_dict['raw_data'][bs_data_index]])
    ax5 = default_histogram(data=user_time, n_bs=n_bs, subplot=ax5, path=path, title='UE time in 1s',
                      save_name='user_time_', binrange=(0, 1))

    # UE bandwidth plot
    user_bw = np.concatenate([x['user_bw'] for x in data_dict['raw_data'][bs_data_index]])
    max_bw = np.round(global_parameters['bs_param']['bw']/global_parameters['bs_param']['n_sectors']).astype(int)
    ax6 = default_histogram(data=user_bw, n_bs=n_bs, subplot=ax6, path=path, title='Bandwidth per UE (MHz)',
                      save_name='bw_user_', binrange=(0, max_bw))

    # Capacity deficit plot
    norm_deficit = np.concatenate([x['norm_deficit'] for x in data_dict['raw_data'][bs_data_index]])
    ax7 = default_histogram(data=norm_deficit, n_bs=n_bs, path=path,
                            title='Normalized Capacity Deficit for ' + str(criteria)+ ' Mbps',
                            save_name='norm_deficit_', binrange=(-1, 1))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + 'Metrics_' + str(n_bs) + ' BS.png')


def plot_curves(name_file, max_iter, bs_list, global_parameters):
    data_dict, path = load_data(name_file=name_file, return_path=True)
    plt.rcParams['font.size'] = '4'
    fig_curve, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=500)
    fig_curve.suptitle('Metrics evolution by BS number - ' + str(max_iter) + ' iterations')

    ax1 = default_curve_plt(subplot=ax1, n_bs_vec=bs_list, data=data_dict['mean_snr'],
                      std=data_dict['std_snr'], xlabel='Number of BSs', title='Average SNIR (dB)')
    ax2 = default_curve_plt(subplot=ax2, n_bs_vec=bs_list, data=data_dict['mean_cap'], std=data_dict['std_cap'],
                      xlabel='Number of BSs', title='Average Capacity (Mbps)')
    ax3 = default_curve_plt(subplot=ax3, n_bs_vec=bs_list, data=data_dict['mean_user_time'], std=data_dict['std_user_time'],
                      xlabel='Number of BSs', title='Average UE time (s)')
    ax4 = default_curve_plt(subplot=ax4, n_bs_vec=bs_list, data=data_dict['mean_user_bw'], std=data_dict['std_user_bw'],
                      xlabel='Number of BSs', title='Average UE BW (MHz)')

    fig_curve.tight_layout()
    plt.savefig(path + 'perf_curves.png')
    plt.rcdefaults()

    title = '% of UE that meets the ' + str(global_parameters['downlink_scheduler']['criteria']) + ' Mbps criteria'
    default_curve_plt(n_bs_vec=bs_list, data=data_dict['total_meet_criteria'], xlabel='Number of BSs', title=title,
                      path=path, save=True, save_name='cap_defict')

    # special plots
    # avg UE that meet the criteria per time slot
    time_shape = global_parameters['macel_param']['time_slots']//global_parameters['macel_param']['time_slot_lngt']
    plt_line = np.zeros(shape=(data_dict['BSs'].__len__(), time_shape))
    for i, bs_data_index in enumerate(data_dict['BSs']):
        plt_line[i] = np.mean(extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='meet_criteria',
                                               bs_data_index=i, concatenate=False), axis=0)

    fig_criteria_it = plt.figure(figsize=(13, 10), dpi=100)
    line_objects = plt.plot(np.array(plt_line.T) / 10)
    plt.title('Average number of UEs that meet the criteria of ' +
              str(global_parameters['downlink_scheduler']['criteria']) + 'Mbps per time slot').set_size(19)
    plt.ylim([0, 100])
    plt.legend(iter(line_objects), data_dict['BSs'], title='nBS', fontsize='x-large')
    plt.xlabel('time slot').set_size(15)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(path + 'meet_criteria_time.png')
    plt.close('all')

    # inactive UEs
    title = 'UEs not connected to the RAN'
    beam_sec_groupings = group_ue(data_dict=data_dict)
    avg_disconn_ues = [np.mean(x['nactive_ue_cnt']) for x in beam_sec_groupings]
    std_disconn_ues = [np.std(x['nactive_ue_cnt']) for x in beam_sec_groupings]
    default_curve_plt(n_bs_vec=bs_list, data=avg_disconn_ues, std=std_disconn_ues, xlabel='Number of BSs', title=title,
                      path=path, save=True, save_name='not_connected_ues')

    # RAN Capacity per time_slot
    # ran_cap = np.zeros(shape=[global_parameters['exec_param']['max_iter'], global_parameters['macel_param']['time_slots']])
    plt_line = np.zeros(shape=(data_dict['BSs'].__len__(), time_shape))
    for bs_data_index, _ in enumerate(data_dict['BSs']):
        ran_cap = np.array(extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='ran_cap_per_time',
                                       bs_data_index=bs_data_index, concatenate=False))
        avg_ran_cap = np.mean(ran_cap, axis=0)
        std_ran_cap = np.std(ran_cap, axis=0)
        plt_line[bs_data_index] = avg_ran_cap

    fig_ran_cap_it = plt.figure(figsize=(13, 10), dpi=100)
    line_objects = plt.plot(np.array(plt_line.T))
    plt.title('Average RAN Capacity per Time Slot (Mbps)').set_size(19)
    plt.ylim([0, 100])
    plt.legend(iter(line_objects), data_dict['BSs'], title='nBS', fontsize='x-large')
    plt.xlabel('time slot').set_size(15)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(path + 'RAN_capacity_time.png')
    plt.close('all')

    # spectral efficiency and RAN throughput
    thgp_tot_avg = []
    thgp_tot_std = []
    sp_eff_tot_avg = []
    sp_eff_tot_std = []
    for bs_index, n_bs in enumerate(data_dict['BSs']):
        raw_cap = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='cap', bs_data_index=bs_index,
                                   concatenate=False)
        bw_tot = n_bs * global_parameters['bs_param']['n_sectors']  # TODO - ARRUMAR A PORRA DO BW NO ARQUIVO PARAM QUE ESTÁ MULTIPLICADO PELO N_SEC
        ran_cap = np.array([np.sum(x) for x in raw_cap])/1000  # capacity in Gbps
        spc_eff = ran_cap/bw_tot

        thgp_tot_avg.append(ran_cap.mean())
        thgp_tot_std.append(ran_cap.std())
        sp_eff_tot_avg.append(spc_eff.mean())
        sp_eff_tot_std.append(spc_eff.std())

    default_curve_plt(n_bs_vec=data_dict['BSs'], data=thgp_tot_avg, std=thgp_tot_std, xlabel='Number of BSs',
                      title='RAN total throughput (Gbps)', path=path, save=True, save_name='ran_throughput')

    default_curve_plt(n_bs_vec=data_dict['BSs'], data=sp_eff_tot_avg, std=sp_eff_tot_std, xlabel='Number of BSs',
                      title='Spectral Efficiency (bits/Hz)', path=path, save=True, save_name='ran_efficiency')


def plot_surfaces(name_file, global_parameters, n_bs=None):
    data_dict = load_data(name_file=name_file)
    if n_bs is None:
        n_bs = data_dict['BSs'][-1]  # picking the last simulation
    bs_data_index = np.where(np.array(data_dict['BSs']) == n_bs)[0][0]

    path = create_subfolder(name_file=name_file, n_bs=n_bs)

    grid = Grid()
    grid.make_grid(lines=global_parameters['roi_param']['grid_lines'],
                   columns=global_parameters['roi_param']['grid_columns'])

    # data coordinates
    bs_coordinates = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='bs_position',
                                                bs_data_index=bs_data_index)
    ue_coordinates = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='ue_position',
                                                bs_data_index=bs_data_index)

    cap = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='cap', bs_data_index=bs_data_index,
                                     calc='avg')
    default_surf_plt(data=cap, grid=grid.grid, coordinates=bs_coordinates, n_bs=n_bs, max_iter=global_parameters['exec_param']['max_iter'],
                     title='Capacity (Mbps)', path=path, save_name='cap_bs_points', save=True)

    snr = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='snr', bs_data_index=bs_data_index,
                                     calc='avg')
    default_surf_plt(data=snr, grid=grid.grid, coordinates=bs_coordinates, n_bs=n_bs,max_iter=global_parameters['exec_param']['max_iter'],
                     title='SNIR (dB)', path=path, save_name='snr_bs_points', save=True)


def default_surf_plt(data, grid, coordinates, n_bs, max_iter, title, path=None, save_name=False, save=False):
    X = coordinates[:, :, :, 0]
    Y = coordinates[:, :, :, 1]
    counter = copy.copy(grid)
    norm = copy.copy(grid)
    norm.fill(np.nan)

    # for i, [x, y] in enumerate(zip(X, Y)):
    for i, [k, l] in enumerate(zip(X,Y)):
        grid[k, l] += data[i]
        counter[k, l] += 1

    norm[counter != 0] = grid[counter != 0]/counter[counter != 0]
    fig, ax1 = plt.subplots(1, dpi=300)
    fig.suptitle(title + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')
    z = ax1.matshow(norm, origin='lower', cmap=cm.Spectral_r)
    fig.colorbar(z, ax=ax1)
    ax1.set(facecolor="black")

    if save:
        if path is not None:
            if save_name is None:
                save_name = title
            fig.savefig(path + save_name + str(n_bs) + ' BS.png')
        else:
            'EXCEÇÃO AQUI'

    return fig


def default_curve_plt(n_bs_vec, data, xlabel, title, subplot=None, std=None, dpi=150, save=False, path=None,
                      save_name=None, show=False):
    # individual plot
    if subplot is None:
        plt.close('all')
        plt.rcdefaults()
        fig, subplot = plt.subplots(dpi=dpi)

    subplot.plot(n_bs_vec, data)
    subplot.grid()


    if std is not None:
        data = np.array(data)
        std = np.array(std)
        subplot.fill_between(n_bs_vec, data + std, data - std, alpha=0.3)
    subplot.set_xlabel(xlabel)
    subplot.set_title(title)

    if save:
        if path is not None:
            if save_name is None:
                save_name = title
            fig.savefig(path + save_name + '.png')
        else:
            'EXCEÇÃO AQUI'

    if show:
        fig.show()

    return subplot


def default_histogram(data, n_bs, path, title,  subplot=None, save_name=None, bins=100, binrange=(0, 140),
                      dpi=100, ylim=1, show_plot=False):
    if save_name is None:
        save_name = title

    # indivial plot
    f1 = plt.figure(8, dpi=dpi)
    sns.histplot(data=data, bins=bins, binrange=binrange, stat='probability')
    plt.title(title)
    plt.ylim(0, ylim)
    if show_plot:
        plt.show()
    plt.savefig(path + save_name + str(n_bs) + ' BS.png')
    plt.close()

    # subplot
    if subplot is not None:
        sns.histplot(data=data, bins=bins, binrange=binrange, stat='probability', ax=subplot)  # seaborn
        # ax1.hist(snr, bins=100, density=True, range=(0,100))  # matplotlib
        subplot.set_title(title)

    return subplot

def default_scatter(x, y, path, title, n_bs, dpi=100, subplot=None, alpha=0.35, s=2, color='purple',
                    save_name=None, show_plot=False, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if save_name is None:
        save_name = title

    # indivial plot
    f1 = plt.figure(8, dpi=dpi)
    sns.scatterplot(x, y, alpha=alpha, s=s, color=color)
    plt.grid()
    # sns.histplot(data=data, bins=bins, binrange=binrange, stat='probability')
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(0, ylim)
    if xlim is not None:
        plt.xlim(0, xlim)
    if show_plot:
        plt.show()
    plt.savefig(path + save_name + str(n_bs) + ' BS.png')
    plt.close()

    # subplot
    if subplot is not None:
        sns.scatterplot(x,y, alpha=alpha, s=s, color=color)
        # sns.histplot(data=data, bins=bins, binrange=binrange, stat='probability', ax=subplot)  # seaborn
        # ax1.hist(snr, bins=100, density=True, range=(0,100))  # matplotlib
        subplot.set_title(title)

    return subplot