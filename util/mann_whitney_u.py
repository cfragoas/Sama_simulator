from scipy.stats import mannwhitneyu
import numpy as np

def compare_dist(data1, data2, param):
    # this function will compare two distributions using mann whitneya and returns if true if the distributions can be
    # considered similar

    check1, check2 = True, True

    # if data1[0]['downlink_results'] is not None:
    #     data1_ = np.concatenate([x['downlink_results']['raw_data_dict']['cap'] for x in data1])
    #     data2_ = np.concatenate([x['downlink_results']['raw_data_dict']['cap'] for x in data2])
    #     stat, p = mannwhitneyu(data1_, data2_)
    #     print('Checking similarity for downlink data:')
    #     check1 = check_similarity(stat=stat, p=p, data1=data1_, data2=data2_)
    # if data1[0]['uplink_results'] is not None:
    #     data1_ = np.concatenate([x['uplink_results']['raw_data_dict']['cap'] for x in data1])
    #     data2_ = np.concatenate([x['uplink_results']['raw_data_dict']['cap'] for x in data2])
    #     stat, p = mannwhitneyu(data1_, data2_)
    #     print('Checking similarity for uplink data:')
    #     check2 = check_similarity(stat=stat, p=p, data1=data1_, data2=data2_)

    if data1[0]['downlink_results'] is not None:
        data1_ = np.concatenate([x['downlink_results']['raw_data_dict'][param] for x in data1])
        data2_ = np.concatenate([x['downlink_results']['raw_data_dict'][param] for x in data2])
        stat, p = mannwhitneyu(data1_, data2_)
        print('Checking similarity for downlink data [' + param + '] :')
        check1 = check_similarity(stat=stat, p=p, data1=data1_, data2=data2_)
    if data1[0]['uplink_results'] is not None:
        data1_ = np.concatenate([x['uplink_results']['raw_data_dict'][param] for x in data1])
        data2_ = np.concatenate([x['uplink_results']['raw_data_dict'][param] for x in data2])
        stat, p = mannwhitneyu(data1_, data2_)
        print('Checking similarity for uplink data [' + param + '] :')
        check2 = check_similarity(stat=stat, p=p, data1=data1_, data2=data2_)

    if check1 & check2:
        return True
    else:
        return False

def check_similarity(stat, p, data1, data2):
    print('statistic = %.3f, p = %.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution\n')
        print('data1 [mean, std]: [' + str(data1.mean()) + ', ' + str(data1.std()) + ']' +
              'data2 [mean, std]: [' + str(data2.mean()) + ', ' + str(data2.std()) + ']')
        return True
    else:
        print('Probably different distributions')
        print('data1 [mean, std]: [' + str(data1.mean()) + ', ' + str(data1.std()) + ']' +
              'data2 [mean, std]: [' + str(data2.mean()) + ', ' + str(data2.std()) + ']')
        return False
