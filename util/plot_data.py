import os, platform, copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_curves(mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw, std_user_bw,
         total_meet_criteria, max_iter, n_bs_vec, criteria, individual=False, path=''):

    if individual:
        # Mean SNIR
        plt.plot(mean_snr)
        plt.title('Mean SNIR')
        plt.savefig()

        # std SNIR
        plt.plot(std_snr)
        plt.title('std SNIR')
        plt.savefig()

        # mean CAP
        plt.plot(mean_cap)
        plt.title('std SNIR')
        plt.savefig()

        # std CAP
        plt.plot(std_cap)
        plt.title('std SNIR')
        plt.savefig()

        # mean user time
        plt.plot(mean_user_time)
        plt.title('std SNIR')
        plt.savefig()

        # std user time
        plt.plot(std_user_time)
        plt.title('std SNIR')
        plt.savefig()

        # mean user bw
        plt.plot(mean_user_bw)
        plt.title('std SNIR')
        plt.savefig()

        # std user bw
        plt.plot(std_user_bw)
        plt.title('std SNIR')
        plt.savefig()

    plt.rcParams['font.size'] = '4'
    fig_curve, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, dpi=500)
    fig_curve.suptitle('Metrics evolution by BS number - ' + str(max_iter) + ' iterations')
    ax1.plot(n_bs_vec, mean_snr)
    ax1.set_xlabel('Number of BS')
    ax1.set_title('Mean SNIR')
    ax2.plot(n_bs_vec, std_snr)
    ax2.set_xlabel('Number of BS')
    ax2.set_title('std SNIR')
    ax3.plot(n_bs_vec, mean_cap)
    ax3.set_xlabel('Number of BS')
    ax3.set_title('Mean Capacity (Mbps)')
    ax4.plot(n_bs_vec, std_cap)
    ax4.set_xlabel('Number of BS')
    ax4.set_title('std Capacity (Mbps)')
    ax5.plot(n_bs_vec, mean_user_time)
    ax5.set_xlabel('Number of BS')
    ax5.set_title('Mean user time (s)')
    ax6.plot(n_bs_vec, std_user_time)
    ax6.set_xlabel('Number of BS')
    ax6.set_title('std user time (s)')
    ax7.plot(n_bs_vec, mean_user_bw)
    ax7.set_xlabel('Number of BS')
    ax7.set_title('Mean user bw (MHz)')
    ax8.plot(n_bs_vec, std_user_bw)
    ax8.set_xlabel('Number of BS')
    ax8.set_title('std user bw (MHz)')
    fig_curve.tight_layout()
    # plt.show()
    plt.savefig(path + 'perf_curves.png')

    plt.close('all')

    plt.rcdefaults()

    fig_perc = plt.figure(1, dpi=150)
    plt.plot(n_bs_vec, total_meet_criteria)
    plt.title('% of UE that meets the ' + str(criteria) + ' Mbps criteria')
    plt.savefig(path + 'total_meet_criteria.png')
    plt.close('all')

def plot_hist(raw_data, path, n_bs, max_iter, criteria):
    #creating subfolder
    path = path + '\\' + str(n_bs) + 'BSs\\'
    if platform.system() == 'Darwin':
        path = path.replace('\\', '/')
    if not os.path.exists(path):
        os.mkdir(path)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, dpi=100, figsize=(40, 30))
    fig.suptitle('Metrics using ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')

    # SNR
    snr = np.concatenate([x['snr'] for x in raw_data])
    sns.histplot(data=snr, bins=100, binrange=(0,140), stat='probability', ax=ax1)
    # ax1.hist(snr, bins=100, density=True, range=(0,100))
    ax1.set_title('SNIR (dB)')
    f1 = plt.figure(8, dpi=150)
    sns.histplot(data=snr, bins=100, binrange=(0,140), stat='probability', kde=True)
    plt.title('SNIR (dB)')
    plt.ylim(0, 0.4)
    # plt.show()
    plt.savefig(path + 'snr_' + str(n_bs) + ' BS.png')


    # CAP
    cap = np.concatenate([x['cap'] for x in raw_data])
    sns.histplot(data=cap, bins=1000, binrange=(0,250), stat='probability', ax=ax2)
    # ax2.hist(cap, bins=1000, density=True, range=(0, 200))
    ax2.set_title('Throughput (Mbps)')
    f2 = plt.figure(3, dpi=150)
    # plt.hist(cap, bins=1000,  density=True, range=(0, 200))
    sns.histplot(data=cap, bins=1000, binrange=(0,250), stat='probability', kde=True)
    plt.title('Throughput (Mbps)')
    plt.ylim(0, 0.4)
    # plt.show()
    plt.savefig(path + 'cap_' + str(n_bs) + ' BS.png')

    # Users p/ BS
    user_bs = np.concatenate([x['user_bs'] for x in raw_data])
    sns.histplot(data=user_bs, bins=100, binrange=(0, 40), stat='probability', ax=ax3)
    # ax3.hist(user_bs, bins=100, density=True, range=(0, 40))
    ax3.set_title('UEs per BS')
    f3 = plt.figure(4, dpi=150)
    # plt.hist(user_bs, bins=100, density=True, range=(0, 40))
    sns.histplot(data=user_bs, bins=100, binrange=(0,800), stat='probability')
    plt.title('Number of UEs per BS')
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(path + 'user_bs_' + str(n_bs) + ' BS.png')

    # Number of active beams
    act_beams = np.concatenate([x['act_beams'] for x in raw_data])
    sns.histplot(data=act_beams, bins=11, binrange=(0, 10), stat='probability', ax=ax4)
    # ax4.hist(act_beams, bins=11, density=True, range=(0, 10))
    ax4.set_title('Act beam p/BS')
    f4 = plt.figure(5, dpi=150)
    # plt.hist(act_beams, bins=11, density=True, range=(0, 10))
    sns.histplot(data=act_beams, bins=11, binrange=(0, 10), stat='probability')
    plt.title('Number of Active beams per BS')
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(path + 'act_beams_' + str(n_bs) + ' BS.png')

    # time per user in 1s
    user_time = np.concatenate([x['user_time'] for x in raw_data])
    sns.histplot(data=user_time, bins=50, binrange=(0, 1), stat='probability', ax=ax5)
    # ax5.hist(user_time, bins=50, density=True, range=(0, 1))
    ax5.set_title('UE time in 1s')
    f5 = plt.figure(6, dpi=150)
    # plt.hist(user_time, bins=50, density=True, range=(0, 1))
    sns.histplot(data=user_time, bins=50, binrange=(0, 1), stat='probability')
    plt.title('UE active time in 1s')
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(path + 'user_time_' + str(n_bs) + ' BS.png')

    # bandwidth per user
    user_bw = np.concatenate([x['user_bw'] for x in raw_data])
    sns.histplot(data=user_bw, bins=20, binrange=(0, 100), stat='probability', ax=ax6)
    # ax6.hist(user_bw, bins=20, density=True, range=(0, 100))
    ax6.set_title('BW p/ UE(MHz)')
    f6 = plt.figure(7, dpi=150)
    # plt.hist(user_bw, bins=20, density=True, range=(0, 100))
    sns.histplot(data=user_bw, bins=20, binrange=(0, 100), stat='probability')
    plt.title('Bandwidth per UE (MHz)')
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(path + 'bw_user_' + str(n_bs) + ' BS.png')

    # deficit plots
    fig_deficit, ((ax7, ax8)) = plt.subplots(1, 2, dpi=150)
    fig_deficit.suptitle('Metrics using ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')

    # capacity deficit
    deficit = np.concatenate([x['deficit'] for x in raw_data])
    sns.histplot(data=deficit, bins=100, binrange=(-criteria, criteria), stat='probability', ax=ax7)
    ax7.set_title('Capacity deficit (Mbps)')
    ax7.set_ylim([0, 1])
    fdeftgd = plt.figure(11, dpi=150)
    sns.histplot(data=deficit, bins=100, binrange=(-criteria, criteria), stat='probability')
    plt.title('Capacity deficit (Mbps)')
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(path + 'deficit_' + str(n_bs) + ' BS.png')

    # normalized capacity deficit
    norm_deficit = np.concatenate([x['norm_deficit'] for x in raw_data])
    sns.histplot(data=norm_deficit, bins=100, binrange=(-1, 1), stat='probability', ax=ax8)
    ax8.set_title('Normalized capacity deficit')
    ax8.set_ylim([0, 1])
    f8 = plt.figure(10, dpi=150)
    sns.histplot(data=norm_deficit, bins=100, binrange=(-1, 1), stat='probability')
    plt.title('Normalized capacity deficit')
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(path + 'norm_deficit_' + str(n_bs) + ' BS.png')

    fig_deficit.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_deficit.savefig(path + 'metrics_deficit_' + str(n_bs) + ' BS.png')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + 'Metrics_' + str(n_bs) + ' BS.png')

    # UEs that meet the requeired criteria
    # special case for meet criteira (not a simplified parameters and not a raw histogram case)
    meet_criteria = []
    meet_criteria.append([x['meet_criteria'] for x in raw_data])
    meet_criteria = np.mean(np.array(meet_criteria[0]), axis=0)
    fig_criteria_it = plt.figure(2, dpi=150)
    plt.plot(meet_criteria)
    plt.title('Mean number of UEs that meet the criteria of ' + str(criteria) + 'Mbps per time slot')
    plt.savefig(path + 'meet_criteria_per_iteration.png')

    plt.close('all')

def plot_surface(grid, position, parameter, path, n_bs, max_iter, plt_name=None):
    # creating subfolder
    path = path + '\\' + str(n_bs) + 'BSs\\'
    if platform.system() == 'Darwin':
        path = path.replace('\\', '/')
    if not os.path.exists(path):
        os.mkdir(path)

    snr_sum = grid
    counter = copy.deepcopy(grid)
    # plot surface
    X = position[:, :, 0]
    Y = position[:, :, 1]
    #X, Y = np.meshgrid(X, Y)

    for i, [x, y] in enumerate(zip(X, Y)):
        for j, [k, l] in enumerate(zip(x,y)):
            snr_sum[k, l] += parameter[i]
            counter[k, l] += 1

    mean_snr = snr_sum/np.where(counter == 0, 1, counter)

    fig1, ax1 = plt.subplots(1, dpi=300)
    fig1.suptitle('Accumulated SNIR ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')
    z = ax1.matshow(snr_sum, origin='lower')
    fig1.colorbar(z, ax=ax1)

    plt.savefig(path + 'accum_cap_surf_' + str(n_bs) + ' BS.png')
    plt.close('all')

    fig2, ax1 = plt.subplots(1, dpi=300)
    fig2.suptitle('Average SNIR ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')
    z = ax1.matshow(mean_snr, origin='lower')
    fig2.colorbar(z, ax=ax1)
    print(parameter.shape[0])
    plt.savefig(path + 'mean_cap_surf_' + str(n_bs) + ' BS.png')

    plt.close('all')

    fig3, ax1 = plt.subplots(1, dpi=300)
    fig3.suptitle('BS distribution ' + str(n_bs) + ' BSs and ' + str(max_iter) + ' iterations')
    z = ax1.matshow(counter, origin='lower')
    fig3.colorbar(z, ax=ax1)

    plt.savefig(path + 'bs_dist_surf_' + str(n_bs) + ' BS.png')
    plt.close('all')