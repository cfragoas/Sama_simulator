import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import copy, warnings

from util.data_management import load_data, create_subfolder, extract_parameter_from_raw, group_ue, ue_relative_index
from make_grid import Grid
from matplotlib import cm
from make_raster import Raster


# The function in this file are used to plot and store the plots for the data on the metrics dictionary
# it is assumed that the data has a particular format and these functions also are made to only work with
# the files generated on these simulations
# There are 3 main functions: histogram plots, curve plots and surface plots. The other ones are special cases derived
# from the main ones.

def response_time():
    pass

def plot_histograms(name_file, max_iter, global_parameters, list_typ, n_index=None):
    # plotting the data on a histogram format - it calls the default histogram function and special ones
    # histograms are made for a specified number of BS

    data_dict = load_data(name_file=name_file)  # loading the data dictionaries

    # to navigate one the dictionary and plot with the correct names
    if list_typ == 'BS':
        xlabel = 'Number of BSs'
        iter_dict_name = 'BSs'
    elif list_typ == 'UE':
        xlabel = 'Number of UEs'
        iter_dict_name = 'UEs'
    else:
        raise ValueError('Wrong or empty list_typ set on plot_histograms')

    if n_index is None:  # if the number of BSs has not been informed, it will pick the last one
        if data_dict['downlink_data'][iter_dict_name]:
        # if data_dict['downlink_data']['BSs']:
        #     n_bs = data_dict['downlink_data']['BSs'][-1]  # picking the last simulation
        #     bs_data_index = np.where(np.array(data_dict['downlink_data']['BSs']) == n_bs)[0][0]
            n_index = data_dict['downlink_data'][iter_dict_name][-1]  # picking the last simulation
            data_index = np.where(np.array(data_dict['downlink_data'][iter_dict_name]) == n_index)[0][0]
        elif data_dict['uplink_data'][iter_dict_name]:
            n_index = data_dict['uplink_data'][iter_dict_name][-1]  # picking the last simulation
            data_index = np.where(np.array(data_dict['uplink_data'][iter_dict_name]) == n_index)[0][0]
        # elif data_dict['uplink_data']['BSs']:
        #     n_bs = data_dict['uplink_data']['BSs'][-1]  # picking the last simulation
        #     bs_data_index = np.where(np.array(data_dict['uplink_data']['BSs']) == n_bs)[0][0]

    path = create_subfolder(name_file=name_file, n_index=n_index, dict_name=iter_dict_name)

    # grouping UEs to beams, sectors and BSs (its equal for uplink and downlink)
    # if data_dict['downlink_data']['BSs']:
    if data_dict['downlink_data'][iter_dict_name]:
        beam_sec_groupings = group_ue(data_dict=data_dict['downlink_data'], data_index=data_index,
                                      iter_dict_name=iter_dict_name)[0]
        rel_index_tables = ue_relative_index(data_dict=data_dict['downlink_data'], data_index=data_index)[0]
    # elif data_dict['uplink_data']['BSs']:
    elif data_dict['uplink_data'][iter_dict_name]:
        beam_sec_groupings = group_ue(data_dict=data_dict['uplink_data'], data_index=data_index,
                                      iter_dict_name=iter_dict_name)[0]
        rel_index_tables = ue_relative_index(data_dict=data_dict['uplink_data'], data_index=data_index)[0]

    # beam capacity histogram (downlink and uplink)
    # if data_dict['downlink_data']['BSs']:
    if data_dict['downlink_data'][iter_dict_name]:
        sec_beam_capacity_hist(data_dict=data_dict['downlink_data'], data_index=data_index, n_index=n_index, path=path,
                               beam_sec_groupings=beam_sec_groupings, grouping_name='ue_per_beam',
                               rel_index_tables=rel_index_tables, subname_plot='downlink', dict_name=iter_dict_name)
    # if data_dict['uplink_data']['BSs']:
    if data_dict['uplink_data'][iter_dict_name]:
        sec_beam_capacity_hist(data_dict=data_dict['uplink_data'], data_index=data_index, n_index=n_index, path=path,
                               beam_sec_groupings=beam_sec_groupings, grouping_name='ue_per_beam',
                               rel_index_tables=rel_index_tables, subname_plot='uplink', dict_name=iter_dict_name)

    # sector capacity histogram (downlink and uplink)
    # if data_dict['downlink_data']['BSs']:
    if data_dict['downlink_data'][iter_dict_name]:
        sec_beam_capacity_hist(data_dict=data_dict['downlink_data'], data_index=data_index, n_index=n_index, path=path,
                               beam_sec_groupings=beam_sec_groupings, grouping_name='ue_per_sector',
                               rel_index_tables=rel_index_tables, subname_plot='downlink', dict_name=iter_dict_name)
    # if data_dict['uplink_data']['BSs']:
    if data_dict['uplink_data'][iter_dict_name]:
        sec_beam_capacity_hist(data_dict=data_dict['uplink_data'], data_index=data_index, n_index=n_index, path=path,
                               beam_sec_groupings=beam_sec_groupings, grouping_name='ue_per_sector',
                               rel_index_tables=rel_index_tables, subname_plot='uplink', dict_name=iter_dict_name)

    # capacity x distance (downlink and uplink)
    # if data_dict['downlink_data']['BSs']:
    if data_dict['downlink_data'][iter_dict_name]:
        dist_x_cap_scatter_plot(data_dict=data_dict['downlink_data'], data_index=data_index, n_index=n_index,
                                rel_index_tables=rel_index_tables, path=path, global_parameters=global_parameters,
                                criteria=global_parameters['downlink_scheduler']['criteria'], subname_plot='downlink',
                                dict_name=iter_dict_name)
    # if data_dict['uplink_data']['BSs']:
    if data_dict['uplink_data'][iter_dict_name]:
        dist_x_cap_scatter_plot(data_dict=data_dict['uplink_data'], data_index=data_index, n_index=n_index,
                                rel_index_tables=rel_index_tables, path=path, global_parameters=global_parameters,
                                criteria=global_parameters['uplink_scheduler']['criteria'], subname_plot='uplink',
                                dict_name=iter_dict_name)

    # multiple basic histogram plots  (downlink and uplink)
    # if data_dict['downlink_data']['BSs']:
    if data_dict['downlink_data'][iter_dict_name]:
        histogram_base_plots(data_dict=data_dict['downlink_data'], data_index=data_index, n_index=n_index,
                             max_iter=max_iter, global_parameters=global_parameters, path=path,
                             criteria=global_parameters['downlink_scheduler']['criteria'], subname_plot='downlink',
                             dict_name=iter_dict_name)
    # if data_dict['uplink_data']['BSs']:
    if data_dict['uplink_data'][iter_dict_name]:
        histogram_base_plots(data_dict=data_dict['uplink_data'], data_index=data_index, n_index=n_index,
                             max_iter=max_iter, global_parameters=global_parameters, path=path,
                             criteria=global_parameters['uplink_scheduler']['criteria'], subname_plot='uplink',
                             dict_name=iter_dict_name)

def plot_curves(name_file, max_iter, list_typ, global_parameters):
    # plotting the data on a curve format - it calls the default curve function and special ones
    # all curves uses the BS count as the x-axis

    data_dict, path = load_data(name_file=name_file, return_path=True)  # loading the data dictionaries
    # setting the default uplink/downlink legend
    if data_dict['downlink_data'] is not None and data_dict['uplink_data'] is not None:
        legend = ['downlink', 'uplink']
    else:
        legend = None

    # to navigate one the dictionary and plot with the correct names
    if list_typ == 'BS':
        xlabel = 'Number of BSs'
        iter_dict_name = 'BSs'
        iter_list = global_parameters['exec_param']['executed_n_bs']
    elif list_typ == 'UE':
        xlabel = 'Number of UEs'
        iter_dict_name = 'UEs'
        iter_list = global_parameters['exec_param']['executed_n_ue']
    else:
        raise ValueError('Wrong or empty list_typ set on plot_curves')

    plt.rcParams['font.size'] = '4'

    # creating a figure for multiple basic plots
    # ================================================================================================================
    fig_curve, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=500)
    fig_curve.suptitle('Metrics evolution by BS number - ' + str(max_iter) + ' iterations')

    ax1 = default_curve_plt(subplot=ax1, n_bs_vec=iter_list,
                            data=[data_dict['downlink_data']['mean_snr'], data_dict['uplink_data']['mean_snr']],
                            std=[data_dict['downlink_data']['std_snr'], data_dict['uplink_data']['std_snr']],
                            xlabel=xlabel, title='Average SNIR (dB)', legend=legend)
    ax2 = default_curve_plt(subplot=ax2, n_bs_vec=iter_list,
                            data=[data_dict['downlink_data']['mean_cap'], data_dict['uplink_data']['mean_cap']],
                            std=[data_dict['downlink_data']['std_cap'], data_dict['uplink_data']['std_cap']],
                            xlabel=xlabel, title='Average Capacity (Mbps)', legend=legend)
    ax3 = default_curve_plt(subplot=ax3, n_bs_vec=iter_list,
                            data=[data_dict['downlink_data']['mean_user_time'], data_dict['uplink_data']['mean_user_time']],
                            std=[data_dict['downlink_data']['std_user_time'], data_dict['uplink_data']['std_user_time']],
                            xlabel=xlabel, title='Average UE time (s)', legend=legend)
    ax4 = default_curve_plt(subplot=ax4, n_bs_vec=iter_list,
                            data=[data_dict['downlink_data']['mean_user_bw'], data_dict['uplink_data']['mean_user_bw']],
                            std=[data_dict['downlink_data']['std_user_bw'], data_dict['uplink_data']['std_user_bw']],
                            xlabel=xlabel, title='Average UE BW (MHz)', legend=legend)

    fig_curve.tight_layout()
    plt.savefig(path + 'perf_curves.png')
    plt.rcdefaults()
    # ================================================================================================================

    # UE that meets the uplink/downlink target capacity per BS number curve
    title = '% of UE that meets the ' + str(global_parameters['downlink_scheduler']['criteria']) + ' Mbps criteria'
    default_curve_plt(n_bs_vec=iter_list,
                      data=[data_dict['downlink_data']['total_meet_criteria'],
                            data_dict['uplink_data']['total_meet_criteria']],
                      xlabel=xlabel, title=title, path=path, save=True, save_name='cap_defict', legend=legend,
                      ymax=1.01)

    # ================================ special plots ===============================
    # avg UE that meet the criteria per time slot - need to separate downlink and uplink for this one
    time_shape = global_parameters['macel_param']['time_slots'] // global_parameters['macel_param']['time_slot_lngt']
    if data_dict['downlink_data']['BSs']:
        if data_dict['downlink_data']['total_meet_criteria']:
            criteria_time_slot_plot(data_dict=data_dict['downlink_data'], time_shape=time_shape, path=path,
                                    subname_plot='downlink', iter_dict_name=iter_dict_name,
                                    criteria=global_parameters['downlink_scheduler']['criteria'])
    if data_dict['uplink_data']['BSs']:
        if data_dict['uplink_data']['total_meet_criteria']:
            criteria_time_slot_plot(data_dict=data_dict['uplink_data'], time_shape=time_shape, path=path,
                                    subname_plot='uplink', iter_dict_name=iter_dict_name,
                                    criteria=global_parameters['uplink_scheduler']['criteria'])

    # inactive UEs - this graphics is not different for uplink/downlink
    if data_dict['downlink_data']['BSs']:
        beam_sec_groupings = group_ue(data_dict=data_dict['downlink_data'], iter_dict_name=iter_dict_name)
    elif data_dict['uplink_data']['BSs']:
        beam_sec_groupings = group_ue(data_dict=data_dict['uplink_data'], iter_dict_name=iter_dict_name)
    avg_disconn_ues = [np.mean(x['nactive_ue_cnt']) for x in beam_sec_groupings]
    std_disconn_ues = [np.std(x['nactive_ue_cnt']) for x in beam_sec_groupings]
    title = 'UEs not connected to the RAN'
    default_curve_plt(n_bs_vec=iter_list, data=avg_disconn_ues, std=std_disconn_ues, xlabel=xlabel, title=title,
                      path=path, save=True, save_name='not_connected_ues')

    # RAN Capacity per time_slot
    if data_dict['downlink_data']['BSs']:
        ran_capacity_time_slot_plot(data_dict=data_dict['downlink_data'], time_shape=time_shape, subname='downlink',
                                    path=path, iter_dict_name=iter_dict_name)
    if data_dict['uplink_data']['BSs']:
        ran_capacity_time_slot_plot(data_dict=data_dict['uplink_data'], time_shape=time_shape, subname='uplink',
                                    path=path, iter_dict_name=iter_dict_name)

    thrpt_speceff_fairness_curve_plot(data_dict=[data_dict['downlink_data'], data_dict['uplink_data']],
                                              n_sectors=global_parameters['bs_param']['n_sectors'],
                                              bw=global_parameters['bs_param']['bw'], subname_plot='',
                                              path=path, legend=legend, iter_dict_name=iter_dict_name, xlabel=xlabel )

    # latency plots
    if data_dict['downlink_data']['BSs']:
        latency_plot(raw_data=data_dict['downlink_data']['raw_data'], iter_list=iter_list, path=path,
                     subname_plot='downlink', xlabel=xlabel)
    if data_dict['uplink_data']['BSs']:
        latency_plot(raw_data=data_dict['uplink_data']['raw_data'], iter_list=iter_list, path=path,
                     subname_plot='uplink', xlabel=xlabel)


def plot_surfaces(name_file, global_parameters, list_typ, n_index=None):
    # plotting the data mapped by coordinate - it calls the default curve function and special ones
    # all maps are plotted for a BS number

    # to navigate one the dictionary and plot with the correct names
    if list_typ == 'BS':
        xlabel = 'Number of BSs'
        iter_dict_name = 'BSs'
    elif list_typ == 'UE':
        xlabel = 'Number of UEs'
        iter_dict_name = 'UEs'
    else:
        raise ValueError('Wrong or empty list_typ set on plot_surfaces')

    data_dict = load_data(name_file=name_file)  # loading the data to be ploted
    if n_index is None:  # if the number of BSs has not been informed, it will pick the last one
        # if data_dict['downlink_data']['BSs']:
        if data_dict['downlink_data'][iter_dict_name]:
            n_index = data_dict['downlink_data'][iter_dict_name][-1]  # picking the last simulation
            # n_bs = data_dict['downlink_data']['BSs'][-1]  # picking the last simulation
        elif data_dict['uplink_data'][iter_dict_name]:
        # elif data_dict['uplink_data']['BSs']:
        #     n_bs = data_dict['uplink_data']['BSs'][-1]  # picking the last simulation
            n_index = data_dict['uplink_data'][iter_dict_name][-1]  # picking the last simulation
    # if data_dict['downlink_data']['BSs']: # finding the index for the chosen n_bs
    #     data_index = np.where(np.array(data_dict['downlink_data']['BSs']) == n_bs)[0][0]
    # elif data_dict['uplink_data']['BSs']:
    #     data_index = np.where(np.array(data_dict['uplink_data']['BSs']) == n_bs)[0][0]
    if data_dict['downlink_data'][iter_dict_name]:  # finding the index for the chosen n_bs
        data_index = np.where(np.array(data_dict['downlink_data'][iter_dict_name]) == n_index)[0][0]
    elif data_dict['uplink_data'][iter_dict_name]:
        data_index = np.where(np.array(data_dict['uplink_data'][iter_dict_name]) == n_index)[0][0]

    path = create_subfolder(name_file=name_file, n_index=n_index, dict_name=iter_dict_name)

#Created by Nicholas the if condition bellow:
    if global_parameters['roi_param']['grid']:
        grid = Grid()
        grid.make_grid(lines=global_parameters['roi_param']['grid_lines'],
                   columns=global_parameters['roi_param']['grid_columns'])
    elif global_parameters['roi_param']['raster']:
        grid = Raster(input_shapefile = global_parameters['roi_param']['input_shapefile'],
        output_raster = global_parameters['roi_param']['output_raster'],
        projection = global_parameters['roi_param']['projection'],
        burner_value = global_parameters['roi_param']['burner_value'],
        pixel_size = global_parameters['roi_param']['pixel_size'],
        no_data_value=parameters['roi_param']['no_data_value']
        )

        grid.rasterize_shapefile()
        grid.make_grid()
        grid.delete_tif_file()
        
    # data coordinates - its is the same for downlink and uplink
    if data_dict['downlink_data']['BSs']:
        bs_coordinates = extract_parameter_from_raw(raw_data=data_dict['downlink_data']['raw_data'],
                                                    parameter_name='bs_position', data_index=data_index)
        ue_coordinates = extract_parameter_from_raw(raw_data=data_dict['downlink_data']['raw_data'],
                                                    parameter_name='ue_position', data_index=data_index)
    elif data_dict['uplink_data']['BSs']:
        bs_coordinates = extract_parameter_from_raw(raw_data=data_dict['uplink_data']['raw_data'],
                                                    parameter_name='bs_position', data_index=data_index)
        ue_coordinates = extract_parameter_from_raw(raw_data=data_dict['uplink_data']['raw_data'],
                                                    parameter_name='ue_position', data_index=data_index)

    # capacity map
    if data_dict['downlink_data']['BSs']:
        cap = extract_parameter_from_raw(raw_data=data_dict['downlink_data']['raw_data'], parameter_name='cap',
                                         data_index=data_index, calc='avg')
        default_surf_plt(data=cap, grid=grid.grid, coordinates=bs_coordinates, n_index=n_index, title='Capacity (Mbps)',
                         max_iter=global_parameters['exec_param']['max_iter'], path=path,
                         save_name='downlink_cap_bs_points', save=True, list_typ=list_typ)
    if data_dict['uplink_data']['BSs']:
        cap = extract_parameter_from_raw(raw_data=data_dict['uplink_data']['raw_data'], parameter_name='cap',
                                         data_index=data_index, calc='avg')
        default_surf_plt(data=cap, grid=grid.grid, coordinates=bs_coordinates, n_index=n_index, title='Capacity (Mbps)',
                         max_iter=global_parameters['exec_param']['max_iter'], path=path,
                         save_name='uplink_cap_bs_points', save=True, list_typ=list_typ)

    # SNIR map
    if data_dict['downlink_data']['BSs']:
        snr = extract_parameter_from_raw(raw_data=data_dict['downlink_data']['raw_data'], parameter_name='snr',
                                         data_index=data_index, calc='avg')
        default_surf_plt(data=snr, grid=grid.grid, coordinates=bs_coordinates, n_index=n_index, title='SNIR (dB)',
                         max_iter=global_parameters['exec_param']['max_iter'], path=path,
                         save_name='downlink_snr_bs_points', save=True, list_typ=list_typ)
    if data_dict['uplink_data']['BSs']:
        snr = extract_parameter_from_raw(raw_data=data_dict['uplink_data']['raw_data'], parameter_name='snr',
                                         data_index=data_index, calc='avg')
        default_surf_plt(data=snr, grid=grid.grid, coordinates=bs_coordinates, n_index=n_index, title='SNIR (dB)',
                         max_iter=global_parameters['exec_param']['max_iter'], path=path,
                         save_name='uplink_snr_bs_points', save=True, list_typ=list_typ)


def default_surf_plt(data, grid, coordinates, n_index, max_iter, title, list_typ, path=None, save_name=False, save=False):
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
    fig.suptitle(title + str(n_index) + ' BSs and ' + str(max_iter) + ' iterations')
    z = ax1.matshow(norm, origin='lower', cmap=cm.Spectral_r)
    fig.colorbar(z, ax=ax1)
    ax1.set(facecolor="black")

    if save:
        if path is not None:
            if save_name is None:
                save_name = title
            fig.savefig(path + save_name + str(n_index) + ' ' + list_typ + '.png')
        else:
            'EXCEPTION HERE'

    return fig


def default_curve_plt(n_bs_vec, data, xlabel, title, subplot=None, std=None, dpi=150, save=False, path=None,
                      save_name=None, show=False, legend=None, ymin=None, ymax=None):

    if data.__len__() == 1 or not isinstance(data[0], list):
        # data = np.array(data)[np.newaxis, :]
        data = [data, []]

    if std is not None:
        if std.__len__() == 1 or not isinstance(std[0], list):
            # std = np.array(std)[np.newaxis, :]
            std = [std, []]

    # individual plot
    if subplot is None:
        plt.close('all')
        plt.rcdefaults()
        fig, subplot = plt.subplots(dpi=dpi)

    for d in data:
        if d:
            subplot.plot(n_bs_vec, d)
    subplot.grid()

    if std is not None:
        for i, s in enumerate(std):
            if s and data[i]:
                d = np.array(data[i])
                s = np.array(s)
                subplot.fill_between(n_bs_vec, d + s, d - s, alpha=0.3)
    subplot.set_xlabel(xlabel)
    subplot.set_title(title)

    if ymin:
        subplot.set_ylim(bottom=ymin)
    if ymax:
        subplot.set_ylim(top=ymax)

    if legend is not None:
        plt.legend(legend, fontsize='x-large')

    if save:
        if path is not None:
            if save_name is None:
                save_name = title
            fig.savefig(path + save_name + '.png')
        else:
            'EXCEPTION HERE'

    if show:
        fig.show()

    return subplot


def default_histogram(data, n_index, path, title,  dict_name, subplot=None, save_name=None, bins=100, binrange=(0, 140), alpha=0.3,
                      dpi=100, ylim=1, show_plot=False):
    if save_name is None:
        save_name = title

    if data.__len__() == 1 or not isinstance(data[0], list):
        data = [data, []]

    # indivial plot
    f1 = plt.figure(8, dpi=dpi)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # ignorings warnings because seaborn is cringe
        for d in data:
            sns.histplot(data=d, bins=bins, binrange=binrange, stat='probability', alpha=alpha)
    plt.title(title)
    plt.ylim(0, ylim)
    if show_plot:
        plt.show()
    plt.savefig(path + save_name + str(n_index) + ' ' + dict_name + '.png')
    plt.close()

    # subplot
    if subplot is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignorings warnings because seaborn is cringe
            for d in data:
                sns.histplot(data=d, bins=bins, binrange=binrange, stat='probability', ax=subplot)  # seaborn
        # ax1.hist(snr, bins=100, density=True, range=(0,100))  # matplotlib
        subplot.set_title(title)

    return subplot


def default_scatter(x, y, path, title, n_index, dict_name, dpi=100, subplot=None, alpha=0.35, s=2, color='purple',
                    save_name=None, show_plot=False, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if save_name is None:
        save_name = title

    if np.array(x).shape.__len__() == 1:
        x = np.array(x)[np.newaxis, :]
    if np.array(y).shape.__len__() == 1:
        y = np.array(y)[np.newaxis, :]

    # indivial plot
    f1 = plt.figure(8, dpi=dpi)
    for i, x_coord in enumerate(x):
        sns.scatterplot(x=x_coord, y=y[i], alpha=alpha, s=s, color=color)
    plt.grid()
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
    plt.savefig(path + save_name + str(n_index) + ' ' + dict_name + '.png')
    plt.close()

    # subplot
    if subplot is not None:
        for i, x_coord in enumerate(x):
            sns.scatterplot(x_coord, y[i], alpha=alpha, s=s, color=color)
        subplot.set_title(title)

    return subplot


def criteria_time_slot_plot(data_dict, time_shape, path,  subname_plot, iter_dict_name, criteria):
    # plt_line = np.zeros(shape=(data_dict['BSs'].__len__(), time_shape))
    plt_line = np.zeros(shape=(data_dict[iter_dict_name].__len__(), time_shape))

    for i, data_index in enumerate(data_dict[iter_dict_name]):
        ue_achieve_ctarg = np.array(extract_parameter_from_raw(raw_data=data_dict['raw_data'],
                                                         parameter_name='meet_criteria', data_index=i,
                                                         concatenate=False))
        # # correcting this metric because of the TDD
        max_reference = ue_achieve_ctarg.max(axis=0)
        last_max = 0
        for index, max_ref in enumerate(max_reference):
            if max_ref < last_max:
                ue_achieve_ctarg[:, index] = ue_achieve_ctarg[:, index-1]
            elif max_ref > last_max:
                last_max = max_ref

        ue_bs_table = extract_parameter_from_raw(raw_data=data_dict['raw_data'],
                                                   parameter_name='ue_bs_table', data_index=i,
                                                   concatenate=False)
        n_ues = np.array([x.shape[0] for x in ue_bs_table])
        y = [x/n_ues[j] for j, x in enumerate(ue_achieve_ctarg)]
        plt_line[i] = np.mean(y, axis=0)

    fig_criteria_it = plt.figure(figsize=(13, 10), dpi=100)
    line_objects = plt.plot((np.array(plt_line.T)) * 100)
    plt.title('Average number of UEs that meet the criteria of ' +
              str(criteria) + 'Mbps per time slot').set_size(19)
    plt.ylim([0, 100])
    plt.legend(iter(line_objects), data_dict[iter_dict_name], title='n' + str(iter_dict_name), fontsize='x-large')
    plt.xlabel('time slot').set_size(15)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(path + subname_plot + '_meet_criteria_time.png')
    plt.close('all')


def ran_capacity_time_slot_plot(data_dict, time_shape, subname, path, iter_dict_name):
    # plt_line = np.zeros(shape=(data_dict['BSs'].__len__(), time_shape))
    plt_line = np.zeros(shape=(data_dict[iter_dict_name].__len__(), time_shape))

    # for bs_data_index, _ in enumerate(data_dict['BSs']):
    for data_index, _ in enumerate(data_dict[iter_dict_name]):
        ran_cap = np.array(extract_parameter_from_raw(raw_data=data_dict['raw_data'],
                                                      parameter_name='ran_cap_per_time', data_index=data_index,
                                                      concatenate=False))
        avg_ran_cap = np.mean(ran_cap, axis=0)
        plt_line[data_index] = avg_ran_cap

    fig_ran_cap_it = plt.figure(figsize=(13, 10), dpi=100)
    line_objects = plt.plot(np.array(plt_line.T))
    plt.title('Average RAN Capacity per Time Slot (Mbps)').set_size(19)
    plt.ylim([0, 100])
    # plt.legend(iter(line_objects), data_dict['BSs'], title='nBS', fontsize='x-large')
    plt.legend(iter(line_objects), data_dict[iter_dict_name], title='n' + iter_dict_name, fontsize='x-large')
    plt.xlabel('time slot').set_size(15)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(path + subname + '_RAN_capacity_time.png')
    plt.close('all')


def thrpt_speceff_fairness_curve_plot(data_dict, n_sectors, bw, subname_plot, path, iter_dict_name, xlabel, legend=None):
    if data_dict.__len__() == 1 or not isinstance(data_dict, list):
        data_dict = [data_dict, []]
    iter_dict_name
    thgp_tot_avg_tot = []
    thgp_tot_std_tot = []
    sp_eff_tot_avg_tot = []
    sp_eff_tot_std_tot = []
    fairness_avg_tot = []
    fairness_spd_tot = []

    for d in data_dict:
        if d[iter_dict_name]:
        # if d['BSs']:
            n_vec = d[iter_dict_name]
            thgp_tot_avg = []
            thgp_tot_std = []
            sp_eff_tot_avg = []
            sp_eff_tot_std = []
            fairness_avg = []
            fairness_spd = []
            for index, n_bs in enumerate(d[iter_dict_name]):
            # for bs_index, n_bs in enumerate(d['BSs']):
                raw_cap = extract_parameter_from_raw(raw_data=d['raw_data'], parameter_name='cap',
                                                     data_index=index, concatenate=False)
                bw_tot = n_bs * n_sectors * bw  # TODO - ARRUMAR A PORRA DO BW NO ARQUIVO PARAM QUE ESTÁ MULTIPLICADO PELO N_SEC
                ran_cap = np.array([np.sum(x) for x in raw_cap]) / 1000  # capacity in Gbps
                spc_eff = ran_cap / bw_tot
                fairness = np.array([((cap.sum())**2)/(cap.shape[0]*(cap**2).sum()) for cap in raw_cap])

                thgp_tot_avg.append(ran_cap.mean())
                thgp_tot_std.append(ran_cap.std())
                sp_eff_tot_avg.append(spc_eff.mean())
                sp_eff_tot_std.append(spc_eff.std())
                fairness_avg.append(fairness.mean())
                fairness_spd.append(fairness.std())


            thgp_tot_avg_tot.append(thgp_tot_avg)
            thgp_tot_std_tot.append(thgp_tot_std)
            sp_eff_tot_avg_tot.append(sp_eff_tot_avg)
            sp_eff_tot_std_tot.append(sp_eff_tot_std)
            fairness_avg_tot.append(fairness_avg)
            fairness_spd_tot.append(fairness_spd)
        else:
            thgp_tot_avg_tot.append([])
            thgp_tot_std_tot.append([])
            sp_eff_tot_avg_tot.append([])
            sp_eff_tot_std_tot.append([])
            fairness_avg_tot.append([])
            fairness_spd_tot.append([])

    default_curve_plt(n_bs_vec=n_vec, data=thgp_tot_avg_tot, std=thgp_tot_std_tot,
                      xlabel=xlabel, title='RAN total throughput (Gbps)', path=path, save=True,
                      save_name=subname_plot + 'ran_throughput', legend=legend)

    default_curve_plt(n_bs_vec=n_vec, data=sp_eff_tot_avg_tot, std=sp_eff_tot_std_tot,
                      xlabel=xlabel, title='Spectral Efficiency (bits/Hz)', path=path, save=True,
                      save_name=subname_plot + 'ran_efficiency', legend=legend)

    default_curve_plt(n_bs_vec=n_vec, data=fairness_avg_tot, std=fairness_spd_tot,
                      xlabel=xlabel, title='Fairness Index', path=path, save=True,
                      save_name=subname_plot + 'fairness_index', legend=legend)


def sec_beam_capacity_hist(data_dict, data_index, n_index, path, beam_sec_groupings, rel_index_tables, subname_plot,
                           grouping_name, dict_name):
    beam_cap = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='cap',
                                          data_index=data_index, concatenate=False)
    # beam capacity calculation
    acc_beam_cap = []
    # for index in range(max_iter):
    for index, groupings in enumerate(beam_sec_groupings['ue_per_beam']):
        for beam_ues in groupings:
            rel_beam_ues = np.array(rel_index_tables[index]['relative_index'])[
                np.isin(np.array(rel_index_tables[index]['ue_bs_index']), beam_ues)]
            acc_beam_cap.extend([beam_cap[index][rel_beam_ues].sum()])

    if grouping_name == 'ue_per_beam':
        title = 'Capacity per beam (Mbps)'
    elif grouping_name == 'ue_per_sector':
        title = 'Capacity per sector (Mbps)'
    else:
        title = 'TITLE ERROR'

    default_histogram(data=acc_beam_cap, n_index=n_index, path=path, title=title,
                      save_name=subname_plot + '_beam_cap', bins=100, binrange=(0, 100), dict_name=dict_name)
    plt.close('all')


def dist_x_cap_scatter_plot(data_dict, data_index, n_index, rel_index_tables, criteria, path, global_parameters,
                            subname_plot, dict_name):
    raw_dist = extract_parameter_from_raw(raw_data=data_dict['raw_data'], parameter_name='dist_map',
                                          data_index=data_index, concatenate=False)
    ue_bs_indexes = extract_parameter_from_raw(raw_data=data_dict['raw_data'],
                                               parameter_name='ue_bs_table', data_index=data_index,
                                               concatenate=False)
    ue_bs_indexes = [np.array(x['bs_index'][x['bs_index'] != -1]) for x in ue_bs_indexes]

    # raw_dist = np.concatenate([
    #     raw_dist[i][ue_bs_indexes[i]][np.array(x['ue_bs_index'])] for i, x in enumerate(rel_index_tables)])
    # raw_dist = np.concatenate([raw_dist[i][ue_bs_indexes[i], np.array(x['ue_bs_index'])] for i, x in enumerate(rel_index_tables)])
    raw_dist = np.concatenate([raw_dist[i][ue_bs_indexes[i], np.array(x['ue_bs_index'])] for i, x in enumerate(rel_index_tables)])
    raw_cap = np.concatenate(extract_parameter_from_raw(raw_data=data_dict['raw_data'],
                                                        parameter_name='cap', data_index=data_index,
                                                        concatenate=False))

    title = 'Capacity x Distance for ' + str(n_index) + dict_name
    max_dist = (np.sqrt(global_parameters['roi_param']['grid_columns'] * global_parameters['roi_param']['grid_lines']) *
                global_parameters['roi_param']['cel_size']) * 1.05/1000
    if criteria is not None:
        default_scatter(x=raw_dist/1000, y=raw_cap, path=path, title=title, n_index=n_index, dpi=100, subplot=None,
                        save_name=subname_plot + '_cap_x_dist', ylabel='Capacity (Mbps)', xlabel='Distance (km)',
                        xlim=max_dist, ylim=criteria*1.2, dict_name=dict_name)
    else:
        default_scatter(x=raw_dist / 1000, y=raw_cap, path=path, title=title, n_index=n_index, dpi=100, subplot=None,
                        save_name=subname_plot + '_cap_x_dist', ylabel='Capacity (Mbps)', xlabel='Distance (km)',
                        xlim=max_dist, dict_name=dict_name)

def histogram_base_plots(data_dict, data_index, n_index, max_iter, global_parameters, criteria, path, subname_plot,
                         dict_name):
    # default basic histogram plots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=100, figsize=(40, 30))
    fig.suptitle('Metrics using ' + str(n_index) + ' BSs and ' + str(max_iter) + ' iterations')

    # SINR plot
    snr = np.concatenate([x['snr'] for x in data_dict['raw_data'][data_index]])
    ax1 = default_histogram(data=snr, n_index=n_index, subplot=ax1, path=path, title='SNIR (dB)',
                            save_name=subname_plot + '_snr_', bins=100, binrange=(-20, 20), dict_name=dict_name)

    # Capacity plot
    cap = np.concatenate([x['cap'] for x in data_dict['raw_data'][data_index]])
    if criteria is not None:
        ax2 = default_histogram(data=cap, n_index=n_index, subplot=ax2, path=path, title='Throughput (Mbps)',
                                save_name=subname_plot + '_cap_', bins=100,
                                binrange=(0, np.ceil(criteria * 1.05).astype(int)), dict_name=dict_name)
    else:
        ax2 = default_histogram(data=cap, n_index=n_index, subplot=ax2, path=path, title='Throughput (Mbps)',
                                save_name=subname_plot + '_cap_', bins=100, dict_name=dict_name)

    # UE per BS plot
    user_bs = np.concatenate([x['user_bs'] for x in data_dict['raw_data'][data_index]])
    # n_ue = global_parameters['macel_param']['n_samples'] * global_parameters['macel_param']['n_centers']
    n_ue = data_dict['UEs'][data_index]
    ax3 = default_histogram(data=user_bs, n_index=n_index, subplot=ax3, path=path, title='Number of UEs per BS',
                            save_name=subname_plot + '_user_bs_', bins=100, binrange=(0, n_ue), dict_name=dict_name)

    # Number of active beams plot
    act_beams = np.concatenate([x['act_beams'] for x in data_dict['raw_data'][data_index]])
    n_beams = global_parameters['bs_param']['n_beams']
    ax4 = default_histogram(data=act_beams, n_index=n_index, subplot=ax4, path=path, title='Number of Active beams per BS',
                            save_name=subname_plot + '_act_beams_', bins=n_beams, binrange=(0, n_beams),
                            dict_name=dict_name)

    # Active UE time plot
    user_time = np.concatenate([x['user_time'] for x in data_dict['raw_data'][data_index]])
    ax5 = default_histogram(data=user_time, n_index=n_index, subplot=ax5, path=path, title='UE time in 1s',
                            save_name=subname_plot + '_user_time_', binrange=(0, 1), dict_name=dict_name)

    # UE bandwidth plot
    user_bw = np.concatenate([x['user_bw'] for x in data_dict['raw_data'][data_index]])
    max_bw = np.round(global_parameters['bs_param']['bw'] / global_parameters['bs_param']['n_sectors']).astype(int)
    ax6 = default_histogram(data=user_bw, n_index=n_index, subplot=ax6, path=path, title='Bandwidth per UE (MHz)',
                            save_name=subname_plot + '_bw_user_', binrange=(0, max_bw), dict_name=dict_name)

    # Capacity deficit plot
    if criteria is not None:
        norm_deficit = np.concatenate([x['norm_deficit'] for x in data_dict['raw_data'][data_index]])
        ax7 = default_histogram(data=norm_deficit, n_index=n_index, path=path,
                                title='Normalized Capacity Deficit for ' + str(criteria) + ' Mbps',
                                save_name=subname_plot + '_norm_deficit_', binrange=(-1, 1), dict_name=dict_name)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + subname_plot + '_Metrics_' + str(n_index) + ' BS.png')

def latency_plot(raw_data, iter_list, path, xlabel, subname_plot=''):
    # -------------------------------------------- latency calculation --------------------------------------------
    avg_avg_latency = np.zeros(shape=iter_list.__len__())
    avg_avg_latency.fill(np.nan)
    std_avg_latency = copy.copy(avg_avg_latency)
    avg_start_latency = copy.copy(avg_avg_latency)
    std_start_latency = copy.copy(avg_avg_latency)
    avg_min_latency = copy.copy(avg_avg_latency)
    std_min_latency = copy.copy(avg_avg_latency)
    avg_max_latency = copy.copy(avg_avg_latency)
    std_max_latency = copy.copy(avg_avg_latency)

    for index, _ in enumerate(iter_list):
        avg_latency = extract_parameter_from_raw(raw_data=raw_data, parameter_name='avg_latency', data_index=index)
        start_latency = extract_parameter_from_raw(raw_data=raw_data, parameter_name='start_latency', data_index=index)
        min_latency = extract_parameter_from_raw(raw_data=raw_data, parameter_name='min_latency', data_index=index)
        max_latency = extract_parameter_from_raw(raw_data=raw_data, parameter_name='max_latency', data_index=index)

        avg_avg_latency[index] = np.nanmean(avg_latency)
        std_avg_latency[index] = np.nanstd(avg_latency)
        avg_start_latency[index] = np.nanmean(start_latency)
        std_start_latency[index] = np.nanstd(start_latency)
        avg_min_latency[index] = np.nanmean(min_latency)
        std_min_latency[index] = np.nanstd(min_latency)
        avg_max_latency[index] = np.nanmean(max_latency)
        std_max_latency[index] = np.nanstd(max_latency)

    legend = ['avg_latency', 'start_latency', 'min_latency', 'max_latency']

    default_curve_plt(n_bs_vec=iter_list,
                      data=[avg_avg_latency.tolist(), avg_start_latency.tolist(), avg_min_latency.tolist(), avg_max_latency.tolist()],
                      std=[std_avg_latency.tolist(), std_start_latency.tolist(), std_min_latency.tolist(), std_max_latency.tolist()],
                      xlabel=xlabel, title='latency (ms)', legend=legend, path=path, save=True, ymin=0,
                      save_name=subname_plot + '_latency')

def spectrum_aux_plot():  # todo - make a function to plot the interferece/SNIR/power spectrums
    pass