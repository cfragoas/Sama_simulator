import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pkl = 'output/08_25_24-18_30_30/08_25_24-18_30_30.pkl'
#
#pkl_df = pd.read_pickle(pkl,compression="infer")
#
#with open(pkl, 'rb') as f:
#    pkl_dict = pickle.load(f)
#
#
##print(type(pkl_df))
##print('------------------------')
##print(type(pkl_dict))
#
##for i, dictionary in enumerate(pkl_dict):
##    print(f"Dictionary {i + 1}:")
##    for key, value in dictionary.items():
##        print(f"  Key: {key}, Value: {value}")
##    print()  # Add a blank line for better readability between dictionaries
#
##for i, dictionary in enumerate(pkl_dict):
#    #print(f"Dictionary {i + 1}: {len(dictionary)} key-value pairs")
#
#for dictionary in pkl_dict:
#    for key, value in dictionary.items():
#        if isinstance(value, dict):  # Check if the value is a dictionary
#            print(f"Keys of '{key}' dictionary: {list(value.keys())}")
#
#print(pkl_dict[0]['downlink_data']['raw_data'])

#Email do Christian:
with open(pkl, 'rb') as f:
    data_dict = pickle.load(f)
    f.close()
   
data_dict = data_dict[0]
for bs_line in data_dict['downlink_data']['raw_data']:
    for iteration in bs_line:
        print(iteration['user_propagation_scenario']) 




