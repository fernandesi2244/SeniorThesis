# Exploratory data analysis on the magnetic field volume parameterization CSV.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../InputData/volume_parameterizations_all_blobs_carved_from_bitmap.csv")

# Columns are ['Filename General', 'Total Magnetic Energy', 'Total Unsigned Current Helicity', 'Total Absolute Net Current Helicity', 'Mean Shear Angle', 'Total Unsigned Volume Vertical Current', 'Twist Parameter Alpha', 'Mean Gradient of Vertical Magnetic Field', 'Mean Gradient of Total Magnetic Field', 'Total Magnitude of Lorentz Force', 'Total Unsigned Magnetic Flux']

# Generate stats for the Total Magnetic Energy column and plot a histogram. Note that we will have 10 histogram subplots for each column. Should be 2x5.
print('Total Magnetic Energy stats:')
print(df['Volume Total Magnetic Energy'].describe())

plt.figure(figsize=(20, 10))
plt.subplot(2, 5, 1)
plt.hist(df['Volume Total Magnetic Energy'], bins=30)
plt.title('Total Magnetic Energy Distribution', fontsize=10)

# Generate stats for the Total Unsigned Current Helicity column and plot a histogram
print('Total Unsigned Current Helicity stats:')
print(df['Volume Total Unsigned Current Helicity'].describe())

plt.subplot(2, 5, 2)
plt.hist(df['Volume Total Unsigned Current Helicity'], bins=30)
plt.title('Total Unsigned Current Helicity Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Total Absolute Net Current Helicity column and plot a histogram
print('Total Absolute Net Current Helicity stats:')
print(df['Volume Total Absolute Net Current Helicity'].describe())

plt.subplot(2, 5, 3)
plt.hist(df['Volume Total Absolute Net Current Helicity'], bins=30)
plt.title('Total Absolute Net Current Helicity Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Mean Shear Angle column and plot a histogram
print('Mean Shear Angle stats:')
print(df['Volume Mean Shear Angle'].describe())

plt.subplot(2, 5, 4)
plt.hist(df['Volume Mean Shear Angle'], bins=30)
plt.title('Mean Shear Angle Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Total Unsigned Volume Vertical Current column and plot a histogram
print('Total Unsigned Volume Vertical Current stats:')
print(df['Volume Total Unsigned Volume Vertical Current'].describe())

plt.subplot(2, 5, 5)
plt.hist(df['Volume Total Unsigned Volume Vertical Current'], bins=30)
plt.title('Total Unsigned Volume Vertical Current Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Twist Parameter Alpha column and plot a histogram
print('Twist Parameter Alpha stats:')
print(df['Volume Twist Parameter Alpha'].describe())

plt.subplot(2, 5, 6)
plt.hist(df['Volume Twist Parameter Alpha'], bins=30)
plt.title('Twist Parameter Alpha Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Mean Gradient of Vertical Magnetic Field column and plot a histogram
print('Mean Gradient of Vertical Magnetic Field stats:')
print(df['Volume Mean Gradient of Vertical Magnetic Field'].describe())

plt.subplot(2, 5, 7)
plt.hist(df['Volume Mean Gradient of Vertical Magnetic Field'], bins=30)
plt.title('Mean Gradient of Vertical Magnetic Field Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Mean Gradient of Total Magnetic Field column and plot a histogram
print('Mean Gradient of Total Magnetic Field stats:')
print(df['Volume Mean Gradient of Total Magnetic Field'].describe())

plt.subplot(2, 5, 8)
plt.hist(df['Volume Mean Gradient of Total Magnetic Field'], bins=30)
plt.title('Mean Gradient of Total Magnetic Field Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Total Magnitude of Lorentz Force column and plot a histogram
print('Total Magnitude of Lorentz Force stats:')
print(df['Volume Total Magnitude of Lorentz Force'].describe())

plt.subplot(2, 5, 9)
plt.hist(df['Volume Total Magnitude of Lorentz Force'], bins=30)
plt.title('Total Magnitude of Lorentz Force Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Generate stats for the Total Unsigned Magnetic Flux column and plot a histogram
print('Total Unsigned Magnetic Flux stats:')
print(df['Volume Total Unsigned Magnetic Flux'].describe())

plt.subplot(2, 5, 10)
plt.hist(df['Volume Total Unsigned Magnetic Flux'], bins=30)
plt.title('Total Unsigned Magnetic Flux Distribution', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# plt.subplots_adjust(wspace=1, hspace=1)

plt.tight_layout()
plt.show()
