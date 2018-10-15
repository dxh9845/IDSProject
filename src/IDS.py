"""
Filename: IDS.py
Description: Produces Scikit Learn moqudel for training / testing
Date: 10/9/2018
Author: Daniel Herzig, Andrew Bertonica
"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class PreprocessResult():
    def __init__(self, regular_dataset, feature_dataset, class_dataset, label_values):
        """
        Create a new preprocess result value.
        
        Arguments:
            regular_dataset {DataFrame} -- the values in the dataframe from the CSV
            feature_dataset {DataFrame} -- the features of the dataframe
            class_dataset {DataFrame} -- the class values fo the dataframe
            label_values {Dict[String]: Array[String]} -- After factorizing the categorical data, store original string values in a dictionary of col_name.
                (label_values[col_name] = [string_values at index of factorized value]
        """

        self.regular_dataset = regular_dataset
        self.feature_dataset = feature_dataset
        self.class_dataset = class_dataset
        self.label_values = label_values

def ReturnNum(col_value):
    if col_value == 'Infinity':
        return np.inf
    elif col_value == 'NaN':
        return np.nan
    else:
        return np.nan

def PrintVerbose(String, Yes):
    """
    Print to output if supplied a True value
    
    Arguments:
        String {String} -- String to print to output
        Yes {boolean} -- Whether or not to print to output
    """

    if Yes:
        print(String)

def ReadAndPreprocess(filename, 
    easyname, 
    col_datatypes=None,
    undersample_benign=False,
    random_sample_pct=None,
    check_weird_cols=False,
    verbose=False):
    """
    Read a CSV file into a Pandas dataframe and preprocess it, splitting into a features dataframe and 
    a class dataframe.
    
    Arguments:
        filename {String} -- the path to the CSV to open 
        easyname {String} -- a name for this dataset to output to console
    
    Keyword Arguments:
        col_datatypes {[ None, dict(string : dtype)]} - A dictionary containing column labels to datatypes. Sets specific columns to specified 
            datatype
        undersample_benign {bool} -- Whether or not to undersample benign entries in hopes of evening dataset (default: {False})
        random_sample_pct {[None, Float]} -- Limit the size of the dataset to a specified percentage (default: {None})
        check_weird_cols {bool} -- Check for columns that contain mixed datatypes (default: {False})
        verbose {bool} -- Print verbose output. (default: {False})
    Returns:
        PreprocessClass -- The Preprocess Result class contaiing (Original CSV Dataframe, Dataframe of Features, Dataframe of Classes, Dictionary of labels for encoded columns)
    """
    
    print("Starting cleaning of {0} dataset".format(easyname))


    # Begin a timer for the dataset
    start = timer()

    if col_datatypes != None:

        PrintVerbose("\tReading CSV with specified datatypes.", verbose)

        df_reg = pd.read_csv(filename, 
            dtype=col_datatypes,
            parse_dates=[' Timestamp'],
            infer_datetime_format=True,
            na_values='NaN', 
            keep_default_na=None)
    else:

        PrintVerbose("\tReading CSV with no specified datatypes.", verbose)

        df_reg = pd.read_csv(filename,
            parse_dates=[' Timestamp'],
            infer_datetime_format=True,
            na_values='NaN', 
            keep_default_na=None)
    
    # Strip the column names of whitespace
    df_reg.columns = df_reg.columns.str.strip()

    # Fill infinities with a negative 2
    df_reg.replace(['Infinity'], -2, inplace=True)
    # Fill NA's with -1s
    df_reg.fillna(-1, inplace=True)

    # One hot encode whether fields are infinity
    df_reg['FlowByteInf'] = df_reg['Flow Bytes/s'] == -2
    df_reg['FlowPacketInf'] = df_reg['Flow Packets/s'] == -2

    # Drop the timestamp field 
    df_reg = df_reg.drop('Timestamp', 1)
    # Drop flow ID
    df_reg = df_reg.drop('Flow ID', 1)

    # Do we want to undersample normal classes to prevent
    # distortion towards normal classes?
    if undersample_benign == True:
        # Calculate number of non-benign samples
        num_attacks  = len(df_reg[df_reg.Label != 'BENIGN'])
        num_benign = len(df_reg[df_reg.Label == 'BENIGN'])

        PrintVerbose("Number of attacks (BEFORE UNDERSAMPLE): {}".format(num_attacks), verbose)
        PrintVerbose("Number of benign (BEFORE UNDERSAMPLE): {}".format(num_benign), verbose)

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

        PrintVerbose("Number of attacks (AFTER UNDERSAMPLE): {}".format(num_attacks), verbose)
        PrintVerbose("Number of benign (AFTER UNDERSAMPLE): {}".format(num_benign), verbose)

    # Do we want to randomly sample our dataframe
    if random_sample_pct != None:
        
        PrintVerbose("\tSpecifying random sample of dataset.", verbose)
        df_reg = df_reg.sample(frac=random_sample_pct, replace=False)

    # Drop our "output" class from the frame
    df_feat = df_reg.drop('Label', 1)
    # Retrieve our output class from the frame
    df_label = df_reg.Label

    encoded_labels = {}

    # Encode the IP addresses
    df_feat['Source IP'], source_uniques = pd.factorize(df_feat['Source IP'])
    df_feat['Destination IP'], destination_uniques = pd.factorize(df_feat['Destination IP'])

    encoded_labels['Source IP'] = source_uniques
    encoded_labels['Destination IP'] = destination_uniques
    
    if check_weird_cols == True:
        for col in df_reg.columns:
            weird = (df_reg[[col]].applymap(type) != df_reg[[col]].iloc[0].apply(type)).any(axis=1)
        
            if len(df_reg[weird]) > 0:
                print(col)

    end = timer()

    print("Completed prepping {0} dataset @ {1:.2f} seconds".format(easyname, end-start))

    return PreprocessResult(df_reg, df_feat, df_label, encoded_labels)

def RunModel(ModelObj, DataDict, ClassifierName):

    ModelObj.fit(DataDict['x_train'], DataDict['y_train'])

    start = timer()
    y_predict = ModelObj.predict(DataDict['x_test'])
    end = timer()

    accuracy = accuracy_score(DataDict['y_test'], y_predict)
    conf_matrix = confusion_matrix(DataDict['y_test'], y_predict, labels=DataDict['class_labels'])
    report_dict = classification_report(
        DataDict['y_test'], y_predict, target_names=DataDict['class_labels'], output_dict=True)

    # Return the different metrics
    return {
        'class_name': ClassifierName,
        'time_to_run': end-start,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'report_dict': report_dict
    }

if __name__ == '__main__':
    
    specify_datatypes = { 'Flow ID': object, ' Source IP': object }

    MondayResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Monday-WorkingHours.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        random_sample_pct=0.5, 
        easyname='Monday',
        verbose=True)

    TuesdayResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Tuesday-workingHours.pcap_ISCX.csv", 
        col_datatypes=specify_datatypes,
        undersample_benign=True,
        easyname='Tuesday',
        verbose=True)

    WednesdayResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=True,
        easyname='Wednesday',
        verbose=True)

    feature_df = pd.concat(
        [ MondayResult.feature_dataset, TuesdayResult.feature_dataset, WednesdayResult.feature_dataset ])
    class_df = pd.concat(
        [MondayResult.class_dataset, TuesdayResult.class_dataset, WednesdayResult.class_dataset ])

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

