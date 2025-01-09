"""
Given a dataset containing information for each active region at the 2D photospheric level
and a datset containing information for the same active regions at the 3D coronal level,
unify the two datasets into a single dataset containing all the information.

Assume the 3D data only contains information for SHARPs with one blob. That way, there is
no ambiguity in which record to associate from the 2D dataset, which contains records for
each blob in a SHARP.
"""

import pandas as pd
import numpy as np
import os
import sys

# Load the 2D and 3D datasets
data2D = pd.read_csv('InputData/3_175_150_100_vector_hourly_with_recent_history_extrapolatedFrom60_AR.csv')
data3D = pd.read_csv('InputData/volume_parameterizations.csv')

# Create a new dataset to store the unified data. Start with the 3D data, and try to find the
# associated records from the 2D data.
full_data = data3D.copy()

# Add the 2D columns: Gradient_00	Gradient_05	Gradient_10	Gradient_15	Gradient_20	Gradient_25	Gradient_30	Gradient_35	Gradient_40	Gradient_45	Gradient_50	Shear_00	Shear_05	Shear_10	Shear_15	Shear_20	Shear_25	Shear_30	Shear_35	Shear_40	Shear_45	Shear_50	Phi	Total Unsigned Current Helicity	Total Photospheric Magnetic Free Energy Density	Total Unsigned Vertical Current	Abs of Net Current helicity	M&X Flare Event Rate	Filename General	Relevant Active Regions	Number of Sunspots	Is Plage	Number of Recent Flares	Degree Distance from Center	Latitude	Carrington Longitude	Stonyhurst Longitude	Magnetic Area	Number of Recent CMEs
# Join on the 'Filename General' column, which both datasets have.
new_cols = ['Gradient_00', 'Gradient_05', 'Gradient_10', 'Gradient_15', 'Gradient_20', 'Gradient_25', 'Gradient_30', 'Gradient_35', 'Gradient_40', 'Gradient_45', 'Gradient_50', 'Shear_00', 'Shear_05', 'Shear_10', 'Shear_15', 'Shear_20', 'Shear_25', 'Shear_30', 'Shear_35', 'Shear_40', 'Shear_45', 'Shear_50', 'Phi', 'Total Unsigned Current Helicity', 'Total Photospheric Magnetic Free Energy Density', 'Total Unsigned Vertical Current', 'Abs of Net Current helicity', 'M&X Flare Event Rate', 'Relevant Active Regions', 'Number of Sunspots', 'Is Plage', 'Number of Recent Flares', 'Degree Distance from Center', 'Latitude', 'Carrington Longitude', 'Stonyhurst Longitude', 'Magnetic Area', 'Number of Recent CMEs']

# Add these columns to the full_data df
for col in new_cols:
    if col == 'Relevant Active Regions':
        full_data[col] = ''
    elif col == 'Is Plage':
        full_data[col] = False
    elif col == 'Number of Recent Flares':
        full_data[col] = -1
    elif col == 'Number of Recent CMEs':
        full_data[col] = -1
    elif col == 'Number of Sunspots':
        full_data[col] = -1
    else:
        full_data[col] = np.nan

full_data['Found Match'] = False # used for removing unmatched records later

# Iterate over each row in full_data and use an apply function to add the other necessary columns
def add_photospheric_data(row):
    # Find the corresponding row in the 2D data
    filename = row['Filename General']
    match = data2D[data2D['Filename General'] == filename]
    if len(match) == 0:
        print(f'!!!\nNo match for {filename}\n!!!')
        return row
    match = match.iloc[0]
    for col in new_cols:
        row[col] = match[col]
    row['Found Match'] = True
    return row

full_data = full_data.apply(add_photospheric_data, axis=1)

# Remove the records that didn't have a match
full_data = full_data[full_data['Found Match']]

# Save the unified dataset
full_data.to_csv('OutputData/UnifiedActiveRegionData.csv', index=False)
