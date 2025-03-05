# Use Flares.json and CMEs.json to add recent flares and CMEs to the 3_175_150_100_vector_daily.csv dataset and 3_175_150_100_vector_hourly.csv dataset.

import pandas as pd
import numpy as np
import json
import datetime
import os
import sys

# Load the datasets
working_on_daily = False
# daily = pd.read_csv('3_175_150_100_vector_daily_extrapolated_fixed.csv')
# hourly = pd.read_csv('InputData/3_175_150_100_vector_hourly.csv')
unified_data = pd.read_csv('../OutputData/UnifiedActiveRegionData_5_15_150_100.csv')

# utility columns
unified_data['datetime'] = unified_data['Filename General'].apply(lambda x: x.split('.')[3])

# Load SEP (CSV), flare (JSON) and CME (JSON) data
SEPs = pd.read_csv('../InputData/SPEs.csv')
flares = pd.read_json('../InputData/Flares.json')
CMEs = pd.read_json('../InputData/CMEs.json')

# TODO: Load TEBBS data here, also push above JSONs to GitHub.
"""
InputData/TEBBS.txt format:

start time 	 peak time 	 end time 	 GOES class 	 Max flux 	 AR 	 duration 	 X(arcsec) 	 Y(arcsec) 	 TMax 	 TMax errmin 	 TMax errmax 	 TMax time 	 EMMax 	 EMMax errmin 	 EMMax errmax 	 EMMax_time 	 TEM_delay 	 temperature_min_flag 	 initials_flag 	 rising_phase_bins 
2002-01-01 08:59:00	2002-01-01 09:10:00	2002-01-01 09:17:00	C4.5	0.00000465798	0	1080			11.7183	10.8413	12.3879	2002-01-01 09:03:51	0.368332	0.316624	0.50056	2002-01-01 09:12:51	540	1	1	153
2002-01-02 03:16:00	2002-01-02 03:23:00	2002-01-02 03:27:00	C5.2	0.00000548935	9751	660	952.6	72.9	14.2162	13.4102	14.7996	2002-01-02 03:20:36	0.284044	0.245136	0.351873	2002-01-02 03:24:00	204	1	1	76

Read this data into a df so we can use it to add more features to the unified data.
"""
# Read TEBBS data
column_names = ['start time', 'peak time', 'end time', 'GOES class', 'Max flux', 'AR', 'duration', 'X(arcsec)', 'Y(arcsec)', 'TMax', 'TMax errmin', 'TMax errmax', 'TMax time', 'EMMax', 'EMMax errmin', 'EMMax errmax', 'EMMax_time', 'TEM_delay', 'temperature_min_flag', 'initials_flag', 'rising_phase_bins']
TEBBS = pd.read_csv('../InputData/TEBBS.txt', sep='\t', names=column_names, skiprows=1)

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

    if row['Filename General'] == 'hmi.sharp_cea_720s.2748.20130515_010000_TAI':
        isTest = True
    else:
        isTest = False

    # Get the number of SEP events that occur within 1 day in the future of the record date (i.e., any records where the record time is
    # between (SEP start time - 1 day) and (SEP end time)). See paper, but this is based on the idea that an AR
    # is almost exactly the same before and after the event actually occurs. Labeling the region as both a positive 
    # and negative example would counteract what the model learns from that data point.

    # Fetch date from 'datetime' column in this format: 20160416_000000_TAI
    record_date = datetime.datetime.strptime(row['datetime'], '%Y%m%d_%H%M%S_TAI')
    # TODO: Revisit this heuristic
    SEPs_in_range = SEPs[(SEPs['endTime'].apply(toDatetime) >= record_date) & (SEPs['beginTime'].apply(toDatetime) <= record_date + datetime.timedelta(days=1))]
    relevant_SEPs = SEPs_in_range[SEPs_in_range['activeRegionNum'].apply(toIntString).isin(associatedARs)]
    # TODO: No other way we can do these associations, right? We have blob lats/lons in case they are useful.
    num_SEPs = len(relevant_SEPs)
    row['Number of SEPs Produced'] = num_SEPs # TODO: Should this match with M&X event rate?

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

    if earliest_begin_time is not None:
        reference_datetime = earliest_begin_time
        # At this point, we've already made the decision to associate the SEP with the region,
        # so the SEP start time should indeed be the reference datetime if we take this association
        # to be true.
    else:
        reference_datetime = record_date

    # Get the flares that occurred within 3 days before of the reference date
    # TODO: Play around with a longer look-ahead window since the look-behind window is already 3 days.
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

    # Find the associated TEBBS entries and get the:
    # - Max flare peak
    # - Min temperature
    # - Median emission measure
    # - Median duration
    # Note that we are currently only doing max/min for flare peak and temperature since that
    # TEBBS plot very closely follows a trailing edge pattern of SEP-resulting flares, whereas
    # the emission measure/duration trend doesn't completely go out to any "edges" of the distribution.

    # Get the TEBBS entries that are associated with the relevant flares
    relevant_TEBBS = TEBBS[TEBBS['AR'].apply(toIntString).isin(associatedARs)]
    relevant_TEBBS = relevant_TEBBS[(relevant_TEBBS['start time'].apply(toDatetime2) >= reference_datetime - datetime.timedelta(days=3))
                                    & (relevant_TEBBS['start time'].apply(toDatetime2) < reference_datetime)]
    relevant_TEBBS = relevant_TEBBS.dropna(subset=['Max flux', 'TMax', 'EMMax', 'duration'])

    if len(relevant_TEBBS) == 0:
        max_flare_peak = 0
        min_temperature = 0
        median_emission_measure = 0
        median_duration = 0
    else:
        # Get the max flare peak, min temperature, median emission measure, and median duration
        max_flare_peak = relevant_TEBBS['Max flux'].max()
        min_temperature = relevant_TEBBS['TMax'].min()
        median_emission_measure = relevant_TEBBS['EMMax'].median()
        median_duration = relevant_TEBBS['duration'].median()
    
    row['Max Flare Peak of Recent Flares'] = max_flare_peak
    row['Min Temperature of Recent Flares'] = min_temperature
    row['Median Emission Measure of Recent Flares'] = median_emission_measure
    row['Median Duration of Recent Flares'] = median_duration

    # Get the number of flares produced, using same timing heuristic as with SEPs
    flares_in_range = flares[(flares['endTime'].apply(toDatetime) >= record_date) & (flares['beginTime'].apply(toDatetime) <= record_date + datetime.timedelta(days=1))]
    flares_in_range = flares_in_range.dropna(subset=['activeRegionNum'])
    relevant_flares = flares_in_range[flares_in_range['activeRegionNum'].apply(toIntString).isin(associatedARs)]
    num_future_flares = len(relevant_flares)
    row['Number of Flares Produced'] = num_future_flares

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

    # Get the number of CMES produced, using same timing heuristic as with SEPs
    CMEs_in_range = CMEs[(CMEs['endTime'].apply(toDatetime) >= record_date) & (CMEs['startTime'].apply(toDatetime) <= record_date + datetime.timedelta(days=1))]
    CMEs_in_range = CMEs_in_range.dropna(subset=['activeRegionNum'])
    relevant_CMEs = CMEs_in_range[CMEs_in_range['activeRegionNum'].apply(toIntString).isin(associatedARs)]
    num_future_CMEs = len(relevant_CMEs)
    row['Number of CMEs Produced'] = num_future_CMEs

    return row

def toDatetime(date):
    return datetime.datetime.strptime(date.strip(), '%Y-%m-%dT%H:%MZ')

def toDatetime2(date):
    return datetime.datetime.strptime(date.strip(), '%Y-%m-%d %H:%M:%S')

# Active region numbers in JSON files automatically get converted to floats.
def toIntString(floatNum):
    return str(int(floatNum))

# rowNum = 0
# daily = daily.apply(add_events, axis=1)
# daily.to_csv('3_175_150_100_vector_daily_extrapolated_with_all_events_prime_all_SEPs_fixed.csv', index=False)

# rowNum = 0
# hourly = hourly.apply(add_events, axis=1)
# hourly.to_csv('3_175_150_100_vector_hourly_with_all_events_prime_all_SEPs.csv', index=False)

rowNum = 0
unified_data = unified_data.apply(add_events, axis=1)
unified_data.to_csv('../OutputData/UnifiedActiveRegionData_with_all_events_including_new_flares_and_TEBBS_fix_and_flare_cme_labels.csv', index=False)
