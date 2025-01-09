# Use Flares.json and CMEs.json to add recent flares and CMEs to the 3_175_150_100_vector_daily.csv dataset and 3_175_150_100_vector_hourly.csv dataset.

import pandas as pd
import numpy as np
import json
import datetime
import os
import sys

# Load the datasets
working_on_daily = True
# daily = pd.read_csv('3_175_150_100_vector_daily_extrapolated_fixed.csv')
hourly = pd.read_csv('InputData/3_175_150_100_vector_hourly.csv')

# Load SEP (CSV), flare (JSON) and CME (JSON) data
SEPs = pd.read_csv('InputData/SPEs.csv')
flares = pd.read_json('InputData/Flares.json')
CMEs = pd.read_json('InputData/CMEs.json')

volume_parameterizations = pd.read_csv('InputData/volume_parameterizations.csv')

# Add the recent flares and CMEs to the datasets
rowNum = 0

# NOTE: Currently includes non-major flares (with class type other than X or M)
def add_events(row):
    global rowNum
    if rowNum % 1000 == 0:
        print("Row number: ", rowNum)
    rowNum += 1

    # Get the relevant active regions for the record
    associatedARs = row['Relevant Active Regions'][1:-1].split(', ')
    # TODO: Temp fix for now until parameterizations are fixed.
    if len(associatedARs) != 1:
        row['Remove'] = True
        return row

    # Cross reference with the volume parameterizations to get the associated row.
    # First, check if the filename is in the volume parameterizations.
    row_filename_base = row['Filename General'].split('/')[-1][:-1]
    # do string equality check on filename general
    associated_volume_row = volume_parameterizations[volume_parameterizations['Filename General'] == row_filename_base]

    # NOTE: CURRENTLY MAKING ASSUMPTION THAT ALL parameterizations are for single-blob SHARPs.
    # TODO: Adjust this to work for multi-blob SHARPs. Probably add AR row when generating volume parameterizations.

    if len(associated_volume_row) == 0:
        row['Remove'] = True
        return row


    # Get the number of SEP events that occur within 1 day in the future of the record date (i.e., any time between SEP start time - 1 day and SEP end time)
    # /share/development/data/drms/MagPy_Shared_Data/TrainingData/hmi.sharp_cea_720s.2748.20130515_000000_TAI.
    if row['Filename General'] == '/share/development/data/drms/MagPy_Shared_Data/TrainingData/hmi.sharp_cea_720s.2748.20130515_000000_TAI.':
        isTest = True
    else:
        isTest = False

    # record_date = datetime.datetime.strptime(row['Filename General'].split('.')[-2], '%Y%m%d_%H%M%S_TAI')
    # Fetch date from 'datetime' column in this format: 20160416_000000_TAI
    record_date = datetime.datetime.strptime(row['datetime'], '%Y%m%d_%H%M%S_TAI')
    SEPs_in_range = SEPs[(SEPs['endTime'].apply(toDatetime) >= record_date) & (SEPs['beginTime'].apply(toDatetime) <= record_date + datetime.timedelta(days=1))]
    relevant_SEPs = SEPs_in_range[SEPs_in_range['activeRegionNum'].apply(toIntString).isin(associatedARs)]
    num_SEPs = len(relevant_SEPs)
    row['Number of SEPs Produced'] = num_SEPs

    if isTest:
        print('For test SEP:')
        print('Record date:', record_date)
        print('SEPs in range:', SEPs_in_range)
        print('Relevant SEPs:', relevant_SEPs)
        print('Number of relevant SEPs:', num_SEPs)

    if num_SEPs != 0:
        earliest_begin_time = relevant_SEPs['beginTime'].apply(toDatetime).min()
    else:
        earliest_begin_time = None

    if working_on_daily and earliest_begin_time is not None:
        reference_datetime = earliest_begin_time
    else:
        reference_datetime = record_date

    # Get the flares that occurred within 3 days before of the reference date
    flares_3_days_before = flares[(flares['beginTime'].apply(toDatetime) >= reference_datetime - datetime.timedelta(days=3)) & (flares['beginTime'].apply(toDatetime) < reference_datetime)]
    
    # Only keep flares that have an 'activeRegionNum' field
    flares_3_days_before = flares_3_days_before.dropna(subset=['activeRegionNum'])
    # Get the flares that belong to the relevant active regions
    relevant_flares = flares_3_days_before[flares_3_days_before['activeRegionNum'].apply(toIntString).isin(associatedARs)]

    # Get the number of flares that belong to the relevant active regions
    num_flares = len(relevant_flares)

    try:
        # Get the maximum class type of the relevant flares
        class_types = relevant_flares['classType']
        # Convert class types to numerical values.
        # Flares are classified by their strength and range from B class (weakest) to C, M and X (strongest). Each step up in letter classification comes with a 10 fold increase in energy. Also, within each letter, the scale is further broken down into numbers from 1 to 9 ... 1 being the weakest in the class and 9 the strongest
        class_type_values = {'B': 10, 'C': 20, 'M': 30, 'X': 40}
        max_class_value = 0
        for class_type in class_types:
            if class_type is None:
                continue
            class_letter = class_type[0]
            class_number = float(class_type[1:])
            class_value = class_type_values[class_letter] + class_number
            if class_value > max_class_value:
                max_class_value = class_value
    except Exception as e:
        print('Error1:', e)
        max_class_value = 0

    row['Number of Recent Flares'] = num_flares
    row['Max Class Type of Recent Flares'] = max_class_value

    # Do the same for CMEs
    CMEs_3_days_before = CMEs[(CMEs['startTime'].apply(toDatetime) >= reference_datetime - datetime.timedelta(days=3)) & (CMEs['startTime'].apply(toDatetime) < reference_datetime)]
    CMEs_3_days_before = CMEs_3_days_before.dropna(subset=['activeRegionNum'])
    relevant_CMEs = CMEs_3_days_before[CMEs_3_days_before['activeRegionNum'].apply(toIntString).isin(associatedARs)]
    num_CMEs = len(relevant_CMEs)

    # Get the maximum product of the halfAngle and speed of the relevant CMEs
    try:            
        # TODO: Make sure this is working for cases where there are more than 1 relevant CME
        halfAngles = [relevant_CMEs.iloc[i]['cmeAnalyses'][0]['halfAngle'] for i in range(len(relevant_CMEs))]
        speeds = [relevant_CMEs.iloc[i]['cmeAnalyses'][0]['speed'] for i in range(len(relevant_CMEs))]
        max_product = 0
        for i in range(len(halfAngles)):
            product = halfAngles[i] * speeds[i]
            if product > max_product:
                max_product = product
    except Exception as e:
        print('Error2:', e)
        max_product = 0
    
    row['Number of Recent CMEs'] = num_CMEs
    row['Max Product of Half Angle and Speed of Recent CMEs'] = max_product

    row['Remove'] = False

    return row

def toDatetime(date):
    return datetime.datetime.strptime(date.strip(), '%Y-%m-%dT%H:%MZ')

# Active region numbers in JSON files automatically get converted to floats.
def toIntString(floatNum):
    return str(int(floatNum))

# TODO: Remove SHARP rows that have duplicate filenames.

rowNum = 0
daily = daily.apply(add_events, axis=1)
daily.to_csv('3_175_150_100_vector_daily_extrapolated_with_all_events_prime_all_SEPs_fixed.csv', index=False)

# rowNum = 0
# hourly = hourly.apply(add_events, axis=1)
# hourly.to_csv('3_175_150_100_vector_hourly_with_all_events_prime_all_SEPs.csv', index=False)
