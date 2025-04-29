"""
Run the SEPVal tests on the optimal models for photospheric, coronal, and combination data.

Negative events list: SEPValidationChallengePhaseIII_NonEvents_v4 - SEPValidationChallengePhaseIII_NonEvents_v4.csv
Positive events list: SEPValidationChallengePhaseIII_SEPevents_v4 - SEPValidationChallengePhaseIII_SEPevents_v4.csv
"""

import os
import sys
import pandas as pd
import numpy as np

import TrainSEPValModel

data_subsets = ['photospheric', 'coronal', 'combination']

# Open the CSV files containing the positive and negative events
non_events = pd.read_csv('../InputData/SEPValidationChallengePhaseIII_NonEvents_v4.csv')
sep_events = pd.read_csv('../InputData/SEPValidationChallengePhaseIII_SEPevents_v4.csv')

non_event_ranges = []
event_ranges = []

# Iterate through each non_event and mark for removal later
for index, row in non_events.iterrows():
    forecast_start_time = row['Event Period Start Time for Continuous Forecast Models (24 hours prior to flare)']
    # Parse the date string into a datetime object (month/day/year hour:minute)
    forecast_start_time = pd.to_datetime(forecast_start_time, format='%m/%d/%Y %H:%M')

    forecast_end_time = row['Event Period End Time for Continuous Forecast Models (~14 hours after flare)']
    forecast_end_time = pd.to_datetime(forecast_end_time, format='%m/%d/%Y %H:%M')

    non_event_ranges.append((forecast_start_time, forecast_end_time))

# Iterate through each sep_event and mark for removal later
for index, row in sep_events.iterrows():
    forecast_start_time = row['Event Period Start Time for Continuous Forecast Models (24 hours prior to start)']
    # Parse the date string into a datetime object (month/day/year hour:minute)
    forecast_start_time = pd.to_datetime(forecast_start_time, format='%m/%d/%Y %H:%M')
    
    forecast_end_time = row['Event Period End Time for Continuous Forecast Models (to peak - can go past if makes sense for your model)']
    forecast_end_time = pd.to_datetime(forecast_end_time, format='%m/%d/%Y %H:%M')

    event_ranges.append((forecast_start_time, forecast_end_time))  

print("Non-event ranges:")
for start_time, end_time in non_event_ranges:
    print(f"Start: {start_time}, End: {end_time}")

print("\nEvent ranges:")
for start_time, end_time in event_ranges:
    print(f"Start: {start_time}, End: {end_time}")

#~~~~~~
# For first set of tests, remove all non_events and sep_events from the training set and evaluate
# the model on these held-out events/non-events.
#~~~~~~
for data_subset in data_subsets:
    unified_data = pd.read_csv('../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv')

    # '%Y%m%d_%H%M%S_TAI'
    unified_data['datetime_dt'] = pd.to_datetime(unified_data['datetime'], format='%Y%m%d_%H%M%S_TAI')

    # Remove rows where the datetime is within the range of any non_event or sep_event, saving the rows for later
    train_data = unified_data.copy()
    test_data = pd.DataFrame(columns=unified_data.columns)

    print('Length of original data:', len(unified_data))

    for start_time, end_time in non_event_ranges:
        # SEP can start 2 days before the record as long as its end time is after the record
        # As a heuristic (with potential data leakage), remove all blobs where there is an
        # SEP whose start time is within 1 day of the blob record datetime.

        filtered_out_train_data = train_data[((train_data['datetime_dt'] - pd.Timedelta(days=1) <= start_time) &
                              (train_data['datetime_dt'] + pd.Timedelta(days=1) >= end_time))]
        train_data = train_data[~((train_data['datetime_dt'] - pd.Timedelta(days=1) <= start_time) &
                       (train_data['datetime_dt'] + pd.Timedelta(days=1) >= end_time))]
        
        test_data = pd.concat([test_data, filtered_out_train_data], ignore_index=True)
        
    for start_time, end_time in event_ranges:
        filtered_out_train_data = train_data[((train_data['datetime_dt'] - pd.Timedelta(days=1) <= start_time) &
                              (train_data['datetime_dt'] + pd.Timedelta(days=1) >= end_time))]
        train_data = train_data[~((train_data['datetime_dt'] - pd.Timedelta(days=1) <= start_time) &
                       (train_data['datetime_dt'] + pd.Timedelta(days=1) >= end_time))]
        
        test_data = pd.concat([test_data, filtered_out_train_data], ignore_index=True)
    
    print('Length of training data:', len(train_data))
    print('Length of test data:', len(test_data))
    
    # Call main function of TrainSEPValModel.py script in same dir
    confusion_matrix = TrainSEPValModel.main(data_subset, train_data, test_data, data_subset)
    
    # Just print the confusion matrix for now
    print(f"Confusion Matrix for {data_subset} data subset:")
    print(confusion_matrix)
    print("\n")
        