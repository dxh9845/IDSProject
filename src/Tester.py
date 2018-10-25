from IDS import ReadAndPreprocess, RunModel
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == '__main__':

    specify_datatypes = {
        'Flow ID': object,
        ' Source IP': object
    }

    anomaly = True

    # MondayResult = ReadAndPreprocess(
    #     r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Monday-WorkingHours.pcap_ISCX.csv",
    #     col_datatypes=specify_datatypes,
    #     random_sample_pct=0.5,
    #     easyname='Monday',
    #     verbose=True)

    TuesdayResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Tuesday-WorkingHours.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=True,
        easyname='Tuesday',
        make_anomaly=anomaly,
        verbose=True)

    WednesdayResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=True,
        easyname='Wednesday',
        make_anomaly=anomaly,
        verbose=True)

    ThursdayMorningResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=True,
        easyname='ThursdayMorning',
        make_anomaly=anomaly,
        verbose=True)

    ThursdayAfternoonResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=True,
        easyname='ThursdayAfternoon',
        make_anomaly=anomaly,
        verbose=True)

    FridayMorningResult = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Morning.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=False,
        easyname='FridayMorning',
        make_anomaly=anomaly,
        verbose=True)

    FridayAfternoonResult1 = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=False,
        easyname='FridayAfternoon1',
        make_anomaly=anomaly,
        verbose=True)

    FridayAfternoonResult2 = ReadAndPreprocess(
        r"ProjectFiles\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        col_datatypes=specify_datatypes,
        undersample_benign=False,
        easyname='FridayAfternoon2',
        make_anomaly=anomaly,
        verbose=True)


    feature_df = pd.concat(
        [
            # MondayResult.feature_dataset, 
            TuesdayResult.feature_dataset,
            WednesdayResult.feature_dataset, 
            ThursdayMorningResult.feature_dataset,
            ThursdayAfternoonResult.feature_dataset,
            FridayMorningResult.feature_dataset,
            FridayAfternoonResult1.feature_dataset,
            FridayAfternoonResult2.feature_dataset
        ])

    class_df = pd.concat(
        [
            # MondayResult.class_dataset,
            TuesdayResult.class_dataset,
            WednesdayResult.class_dataset,
            ThursdayMorningResult.class_dataset,
            ThursdayAfternoonResult.class_dataset,
            FridayMorningResult.class_dataset,
            FridayAfternoonResult1.class_dataset,
            FridayAfternoonResult2.class_dataset
        ]
    )

    # The number of records
    num_records = len(class_df)
    num_columns = len(feature_df.columns)

    print("There are {} records and {} columns after preprocessing.".format(num_records, num_columns))

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, class_df, test_size=0.33)

    train_dict = { 
        'x_train': x_train, 
        'x_test': x_test, 
        'y_train': y_train, 
        'y_test': y_test,
    }

    if not anomaly:
        # Turn the categories into numbers to keep track
        # class_df, label_uniques = pd.factorize(class_df)
        label_uniques = class_df.unique()
    else:
        label_uniques = class_df.unique()
        # if (label_uniques[0] == 0):
        #     label_uniques[0] = "BENIGN"
        #     label_uniques[1] = "ATTACK"
        # else:
        #     label_uniques[1] = "BENIGN"
        #     label_uniques[0] = "ATTACK"
    
    train_dict['class_labels'] = label_uniques

    ResultOne = RunModel(
        RandomForestClassifier(class_weight='balanced'),
        train_dict,
        'Random Forest (Balanced)'
    )
    
    ResultTwo = RunModel(
        RandomForestClassifier(),
        train_dict,
        'Random Forest (Imbalanced)'
    )

    ResultThree = RunModel(
        GaussianNB(),
        train_dict,
        'Naive Bayes (Gaussian)'
    )

    ResultFour = RunModel(
        AdaBoostClassifier(),
        train_dict,
        'Ada Boost'
    )

    ResultFive = RunModel(
        KNeighborsClassifier(weights='uniform'),
        train_dict,
        'K Nearest Neighbors (Uniform Weights)'
    )

    ResultSix = RunModel(
        KNeighborsClassifier(weights='distance'),
        train_dict,
        'K Nearest Neighbors (Weighted Distance)'
    )

    # ResultList = [ResultFive]
    ResultList = [ResultOne, ResultTwo, ResultThree, ResultFour, ResultFive, ResultSix]

    rawdata = defaultdict(list)

    for result in ResultList:
        rawdata['Class Name'].append(result['class_name'])
        rawdata['Accuracy'].append(result['accuracy'])
        rawdata['Confusion Matrix'].append(result['conf_matrix'])
        rawdata['Precision'].append(result['report_dict']['weighted avg']['precision'])
        # rawdata['Recall'].append(result['report_dict']['weighted avg']['recall'])
        rawdata['F1-Score'].append(result['report_dict']['weighted avg']['f1-score'])
        rawdata['Training Time'].append(result['time_to_train'])
        rawdata['Testing Time'].append(result['time_to_test'])


    np.save('results.npy', rawdata)
    resultdf = pd.DataFrame.from_dict(rawdata).drop(columns=['Confusion Matrix'])
    resultdf.to_csv('mydata.csv', index=False, header=True, columns=[
                    'Class Name', 'Training Time', 'Testing Time', 'Precision', 'F1-Score', 'Accuracy'])


