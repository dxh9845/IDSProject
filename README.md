Requires you unzip the files to src/ProjectFiles. In the future, will move this to env variables 

# Install requirements
```
pip install -r requirements.txt
```

## REMINDER: Must unzip the GeneratedLabelledFlows.zip to src/ProjectFiles.

The directory structure should look like: 

```
└───src
    ├───ProjectFiles
    ├───GeneratedLabelledFlows
    │   └───TrafficLabelling
    │           Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    │           Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    │           Friday-WorkingHours-Morning.pcap_ISCX.csv
    │           Monday-WorkingHours.pcap_ISCX.csv
    │           Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    │           Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    │           Tuesday-WorkingHours.pcap_ISCX.csv
    │           Wednesday-workingHours.pcap_ISCX.csv
    ├───IDS.py
    └───Tester.py
```

After running Tester.py, two files will be outputted: mydata.csv and results.npy. mydata.csv outputs a CSV table of the run results. results.npy contains the a DataFrame dictionary that contains all the results contained within the CSV table, as well as confusion matrices for the different classifiers. 

# Run 
```
cd src/
python Tester.py
```