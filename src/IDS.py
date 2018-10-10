"""
Filename: IDS.py
Description: Produces Scikit Learn model for training / testing
Date: 10/9/2018
Author: Daniel Herzig, Andrew Bertonica
"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

def ReturnNum(col_value):
    if col_value == 'Infinity':
        return np.inf
    elif col_value == 'NaN':
        return np.nan
    else:
        return np.nan

def ReadAndPreprocess(filename, easyname, check_weird_cols=False):
    # Begin a timer for the Monday dataset
    start = timer()

    # Benign traffic to normalize
    df_reg = pd.read_csv(filename)
    # Strip the column names of whitespace
    df_reg.columns = df_reg.columns.str.strip()

    # Perform ternary operations on columns which don't comply with dtype
    # Fill infinities with the largest number possible
    df_reg.replace(['Infinity'], np.nan_to_num(np.inf), inplace=True)
    # Fill NA's with 0s
    df_reg.fillna(0, inplace=True)

    # Drop our "output" class from the frame
    df_feat = df_reg.drop('Label', 1)
    # Retrieve our output class from the frame
    df_label = df_reg.Label

    end = timer()

    print("Completed reading {0} dataset @ {1:.2f} seconds".format(easyname, end-start))

    if check_weird_cols:
        for col in df_reg.columns:
            weird = (df_reg[[col]].applymap(type) != df_reg[[col]].iloc[0].apply(type)).any(axis=1)
        
            if len(df_reg[weird]) > 0:
                print(col)

    return (df_reg, df_feat, df_label)

if __name__ == '__main__':
    monday_reg, monday_features, monday_labels = ReadAndPreprocess(r"ProjectFiles\MachineLearningCSV\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv", easyname='Monday')
    
    tuesday_reg, tuesday_features, tuesday_labels = ReadAndPreprocess(r"ProjectFiles\MachineLearningCSV\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv", easyname='Tuesday')
