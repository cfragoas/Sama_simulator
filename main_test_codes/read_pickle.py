import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pkl = 'output/03_14_24-22_13_03/03_14_24-22_13_03.pkl'

pkl_df = pd.read_pickle(pkl,compression="infer")

with open(pkl, 'rb') as f:
    pkl_dict = pickle.load(f)


print(type(pkl_df))
print('------------------------')
print(type(pkl_dict))

#for i, dictionary in enumerate(pkl_dict):
#    print(f"Dictionary {i + 1}:")
#    for key, value in dictionary.items():
#        print(f"  Key: {key}, Value: {value}")
#    print()  # Add a blank line for better readability between dictionaries

#for i, dictionary in enumerate(pkl_dict):
    #print(f"Dictionary {i + 1}: {len(dictionary)} key-value pairs")