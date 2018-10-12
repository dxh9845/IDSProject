"""
Filename: IDS.py
Description: Produces Scikit Learn model for training / testing
Date: 10/9/2018
Author: Daniel Herzig, Andrew Bertonica
"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def ReturnNum(col_value):
    if col_value == 'Infinity':
        return np.inf
    elif col_value == 'NaN':
        return np.nan
    else:
        return np.nan

def ReadAndPreprocess(filename, easyname, undersample_benign=False, random_sample_pct=None, check_weird_cols=False):
    """
    Read a CSV file into a Pandas dataframe and preprocess it, splitting into a features dataframe and 
    a class dataframe.
    
    Arguments:
        filename {String} -- the path to the CSV to open 
        easyname {String} -- a name for this dataset to output to console
    
    Keyword Arguments:
        undersample_benign {bool} -- Whether or not to undersample benign entries in hopes of evening dataset (default: {False})
        random_sample_pct {[None, Float]} -- Check (default: {None})
        check_weird_cols {bool} -- Check for columns that contain mixed datatypes (default: {False})
    
    Returns:
        [tuple] -- A tuple of (Original CSV Dataframe, Dataframe of Features, Dataframe of Classes)
    """

    # Begin a timer for the Monday dataset
    start = timer()

    df_reg = pd.read_csv(filename)
    # Strip the column names of whitespace
    df_reg.columns = df_reg.columns.str.strip()

    # Fill infinities with the largest number possible
    known_irregularities = ['Flow Bytes/s', 'Flow Packets/s']
    df_reg.replace(['Infinity'], np.nan_to_num(np.float32('inf')), inplace=True)
    # Fill NA's with -1s
    df_reg.fillna(-1, inplace=True)

    # Do we want to undersample normal classes to prevent
    # distortion towards normal classes?
    if undersample_benign == True:
        # Calculate number of non-benign samples
        num_attacks  = len(df_reg[df_reg.Label != 'BENIGN'])
        num_benign = len(df_reg[df_reg.Label == 'BENIGN'])

        print("Number of attacks (BEFORE UNDERSAMPLE): {}".format(num_attacks))
        print("Number of benign (BEFORE UNDERSAMPLE): {}".format(num_benign))

        # Get the indices of benign samples
        benign_indices = df_reg[df_reg.Label == 'BENIGN'].index

        # Create a random sample
        random_benign_indices = np.random.choice(benign_indices, num_attacks, replace=False)

        # Find the indices of attacks
        attack_indices = df_reg[df_reg.Label != 'BENIGN'].index 

        # Concatenate the fraud and non-fraud indices
        under_sample_indices = np.concatenate([attack_indices, random_benign_indices])

        # Set the df to the undersample
        df_reg = df_reg.loc[under_sample_indices]

        num_attacks = len(df_reg[df_reg.Label != 'BENIGN'])
        num_benign = len(df_reg[df_reg.Label == 'BENIGN'])

        print("Number of attacks (AFTER UNDERSAMPLE): {}".format(num_attacks))
        print("Number of benign (AFTER UNDERSAMPLE): {}".format(num_benign))

    # Do we want to randomly sample our dataframe
    if random_sample_pct != None:
        df_reg = df_reg.sample(frac=random_sample_pct, replace=False)

    # Drop our "output" class from the frame
    df_feat = df_reg.drop('Label', 1)
    # Retrieve our output class from the frame
    df_label = df_reg.Label

    if check_weird_cols == True:
        for col in df_reg.columns:
            weird = (df_reg[[col]].applymap(type) != df_reg[[col]].iloc[0].apply(type)).any(axis=1)
        
            if len(df_reg[weird]) > 0:
                print(col)

    end = timer()

    print("Completed prepping {0} dataset @ {1:.2f} seconds".format(easyname, end-start))

    return (df_reg, df_feat, df_label)

if __name__ == '__main__':
    monday_reg, monday_features, monday_labels = ReadAndPreprocess(
        r"ProjectFiles\MachineLearningCSV\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv", random_sample_pct=0.5, easyname='Monday')

    tuesday_reg, tuesday_features, tuesday_labels = ReadAndPreprocess(
        r"ProjectFiles\MachineLearningCSV\MachineLearningCVE\Tuesday-workingHours.pcap_ISCX.csv", undersample_benign=True, easyname='Tuesday')
    
    wednesday_reg, wed_features, wed_labels = ReadAndPreprocess(
        r"ProjectFiles\MachineLearningCSV\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv", undersample_benign=True, easyname='Wednesdaya')

    feature_df = pd.concat([monday_features, tuesday_features, wed_features])
    class_df = pd.concat([monday_labels, tuesday_labels, wed_labels])

    x_train, x_test, y_train, y_test = train_test_split(feature_df, class_df, test_size=0.33)

    model = RandomForestClassifier(class_weight='balanced') 
    # model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    # Predict on training set
    print("Unique: {}".format(np.unique(y_predict)))
    
    print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
    CM = confusion_matrix(y_test, y_predict)

    print("FP: {}".format(CM[0][1]))
    print("FN: {}".format(CM[1][0]))
    print("Accuracy score: {}".format(accuracy_score(y_test.values, y_predict)))

