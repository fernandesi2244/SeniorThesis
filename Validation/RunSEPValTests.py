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

data_subsets = ['photospheric', 'coronal', 'numeric']

# Open the CSV files containing the positive and negative events
non_events = pd.read_csv('../InputData/SEPValidationChallengePhaseIII_NonEvents_v4.csv')
sep_events = pd.read_csv('../InputData/SEPValidationChallengePhaseIII_SEPevents_v4.csv')

non_event_ranges = []
non_event_ars = []

event_ranges = []
event_ars = []

# Iterate through each non_event and mark for removal later
for index, row in non_events.iterrows():
    forecast_start_time = row['Event Period Start Time for Continuous Forecast Models (24 hours prior to flare)']
    # Parse the date string into a datetime object (month/day/year hour:minute)
    forecast_start_time = pd.to_datetime(forecast_start_time, format='%m/%d/%Y %H:%M')

    forecast_end_time = row['Event Period End Time for Continuous Forecast Models (~14 hours after flare)']
    forecast_end_time = pd.to_datetime(forecast_end_time, format='%m/%d/%Y %H:%M')

    ar_num = row['Active Region']
    try:
        ar_num = int(ar_num)
    except ValueError:
        ar_num = None

    non_event_ranges.append((forecast_start_time, forecast_end_time))
    non_event_ars.append(ar_num)

# Iterate through each sep_event and mark for removal later
for index, row in sep_events.iterrows():
    forecast_start_time = row['Event Period Start Time for Continuous Forecast Models (24 hours prior to start)']
    # Parse the date string into a datetime object (month/day/year hour:minute)
    forecast_start_time = pd.to_datetime(forecast_start_time, format='%m/%d/%Y %H:%M')
    
    forecast_end_time = row['Event Period End Time for Continuous Forecast Models (to peak - can go past if makes sense for your model)']
    forecast_end_time = pd.to_datetime(forecast_end_time, format='%m/%d/%Y %H:%M')

    ar_num = row['AR']
    try:
        ar_num = int(ar_num)
    except ValueError:
        ar_num = None

    event_ranges.append((forecast_start_time, forecast_end_time))
    event_ars.append(ar_num)

print('Number of non-None ARs in non-events:', len([ar for ar in non_event_ars if ar is not None]))
print('Number of non-None ARs in events:', len([ar for ar in event_ars if ar is not None]))

# print("Non-event ranges:")
# for start_time, end_time in non_event_ranges:
#     print(f"Start: {start_time}, End: {end_time}")

# print("\nEvent ranges:")
# for start_time, end_time in event_ranges:
#     print(f"Start: {start_time}, End: {end_time}")

#~~~~~~
# For first set of tests, remove all non_events and sep_events from the training set and evaluate
# the model on these held-out events/non-events.
#~~~~~~
unified_data = pd.read_csv('../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv')

# '%Y%m%d_%H%M%S_TAI'
unified_data['datetime_dt'] = pd.to_datetime(unified_data['datetime'], format='%Y%m%d_%H%M%S_TAI')
unified_data['Most Probable AR Num'] = unified_data['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])

# Remove rows where the datetime is within the range of any non_event or sep_event, saving the rows for later
train_data = unified_data.copy()
test_data = pd.DataFrame(columns=unified_data.columns)

print('Length of original data:', len(unified_data))

for i, (start_time, end_time) in enumerate(non_event_ranges):
    # SEP can start 2 days before the record as long as its end time is after the record
    # As a heuristic (with potential data leakage), remove all blobs whose record time is
    # between the start time and the end time + 2 days.

    new_test_data = train_data[((train_data['datetime_dt'] >= start_time) &
                            (train_data['datetime_dt'] <= end_time + pd.Timedelta(days=2)))]
    train_data = train_data[~((train_data['datetime_dt'] >= start_time) &
                    (train_data['datetime_dt'] <= end_time + pd.Timedelta(days=2)))]
    test_data = pd.concat([test_data, new_test_data], ignore_index=True)

    # Also remove all blobs whost most probable AR number is in the non-event list
    if non_event_ars[i] is not None:
        new_test_data = train_data[train_data['Most Probable AR Num'] == str(non_event_ars[i])]
        train_data = train_data[train_data['Most Probable AR Num'] != str(non_event_ars[i])]
        test_data = pd.concat([test_data, new_test_data], ignore_index=True)
    
for i, (start_time, end_time) in enumerate(event_ranges):
    new_test_data = train_data[((train_data['datetime_dt'] >= start_time) &
                            (train_data['datetime_dt'] <= end_time + pd.Timedelta(days=2)))]
    train_data = train_data[~((train_data['datetime_dt'] >= start_time) &
                    (train_data['datetime_dt'] <= end_time + pd.Timedelta(days=2)))]
    test_data = pd.concat([test_data, new_test_data], ignore_index=True)

    # Also remove all blobs whose most probable AR number is in the event list
    if event_ars[i] is not None:
        new_test_data = train_data[train_data['Most Probable AR Num'] == str(event_ars[i])]
        train_data = train_data[train_data['Most Probable AR Num'] != str(event_ars[i])]
        test_data = pd.concat([test_data, new_test_data], ignore_index=True)

print('Length of training data:', len(train_data))
print('Length of test data:', len(test_data))

for data_subset in data_subsets:
    # Call main function of TrainSEPValModel.py script in same dir
    train_data_deep = train_data.copy()
    test_data_deep = test_data.copy()
    
    # Create output directory specific to this data subset
    output_dir = f'results/sep_model_results_{data_subset}'
    
    confusion_matrix = TrainSEPValModel.main(data_subset, train_data_deep, test_data_deep, output_dir)
    
    # Just print the confusion matrix for now
    print(f"Confusion Matrix for {data_subset} data subset:")
    print(confusion_matrix)
    print("\n")
