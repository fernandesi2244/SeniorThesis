# Analyze the following columns for outliers and remove them:
# ['Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 'Volume Total Unsigned Magnetic Flux', 'Gradient_00', 'Gradient_10', 'Gradient_30', 'Gradient_50', 'Shear_00', 'Shear_10', 'Shear_30', 'Shear_50', 'Phi', 'Total Unsigned Current Helicity', 'Total Photospheric Magnetic Free Energy Density', 'Total Unsigned Vertical Current', 'Abs of Net Current helicity', 'Number of Sunspots', 'Number of SEPs Produced', 'Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 'Max Product of Half Angle and Speed of Recent CMEs', 'Stonyhurst Longitude']

import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt

# TODO: Retry outlier detectio methods with less-restrictive dataset. Too many values are 0 right now.
# Also, we seem to be removing lots of good values just because they're on the tail end of the distribution.
# See if there are any ways to deal with this appropriately.


file_path = '../OutputData/UnifiedActiveRegionData_with_all_events.csv'
df = pd.read_csv(file_path)

columns = ['Latitude', 'Carrington Longitude', 'Stonyhurst Longitude', 'Volume Total Magnetic Energy', 'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 'Volume Total Unsigned Magnetic Flux', 'Gradient_00', 'Gradient_10', 'Gradient_30', 'Gradient_50', 'Shear_00', 'Shear_10', 'Shear_30', 'Shear_50', 'Phi', 'Total Unsigned Current Helicity', 'Total Photospheric Magnetic Free Energy Density', 'Total Unsigned Vertical Current', 'Abs of Net Current helicity', 'Number of Sunspots', 'Number of SEPs Produced', 'Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 'Max Product of Half Angle and Speed of Recent CMEs', 'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares']

df_inclusion_mask = pd.Series([True] * len(df), index=df.index)

# Analyze the above columns for outliers and remove them:
for column in columns:
    # If the column data span multiple orders of magnitude, first log-transform the data
    column_to_nearest_int = df[column].apply(lambda x: np.round(x))
    if (np.max(column_to_nearest_int) + 1) / (np.min(column_to_nearest_int) + 1) >= 1000:
        print('Log-transforming column:', column)
        df[column] = np.log(df[column] + 1)
    
    # clf = IsolationForest(contamination=0.0005)
    # clf.fit(df[[column]])
    # outliers = clf.predict(df[[column]])
    # print(f'Number of outliers for column {column}: {np.sum(outliers == -1)}')

    # IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    print(f'Number of outliers for column {column}: {outliers.sum()}')

    df_inclusion_mask = df_inclusion_mask & ~outliers

    # z_scores = np.abs(stats.zscore(df[[column]]))
    # threshold = 5
    # outliers = np.where(z_scores > threshold)[0]
    # print(f'Number of outliers for column {column}: {len(outliers)}')

    # Plot the column data to visualize the outliers
    plt.figure()
    plt.hist(df[column], bins=100)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.show()

df = df[df_inclusion_mask]
# df.to_csv(file_path, index=False)

# This file is just for distribution visualization purposes right now because the
# current outlier detection method does not work well with skewed data.
