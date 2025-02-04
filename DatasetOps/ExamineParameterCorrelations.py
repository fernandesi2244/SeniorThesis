"""
Find correlations between the volume and photospheric versions of parameters given by the following pairs:
- Volume Total Unsigned Current Helicity vs. Total Unsigned Current Helicity
- Volume Total Absolute Net Current Helicity vs. Abs of Net Current helicity
- Volume Total Unsigned Volume Vertical Current vs. Total Unsigned Vertical Current
- Volume Total Unsigned Magnetic Flux vs. Phi
- Volume Mean Gradient of Total Magnetic Field vs. Gradient_10

Also plot the correlations.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

file_path = '../OutputData/UnifiedActiveRegionData.csv'
df = pd.read_csv(file_path)

# Volume Total Unsigned Current Helicity vs. Total Unsigned Current Helicity
print('Volume Total Unsigned Current Helicity vs. Total Unsigned Current Helicity')
print('-' * 80)
print('Correlation:', df['Volume Total Unsigned Current Helicity'].corr(df['Total Unsigned Current Helicity']))
print()
plt.scatter(df['Volume Total Unsigned Current Helicity'], df['Total Unsigned Current Helicity'])
plt.xlabel('Volume Total Unsigned Current Helicity')
plt.ylabel('Total Unsigned Current Helicity')
plt.title('Volume Total Unsigned Current Helicity vs. Total Unsigned Current Helicity')
plt.show()

# Volume Total Absolute Net Current Helicity vs. Abs of Net Current helicity
print('Volume Total Absolute Net Current Helicity vs. Abs of Net Current helicity')
print('-' * 80)
print('Correlation:', df['Volume Total Absolute Net Current Helicity'].corr(df['Abs of Net Current helicity']))
print()
plt.scatter(df['Volume Total Absolute Net Current Helicity'], df['Abs of Net Current helicity'])
plt.xlabel('Volume Total Absolute Net Current Helicity')
plt.ylabel('Abs of Net Current helicity')
plt.title('Volume Total Absolute Net Current Helicity vs. Abs of Net Current helicity')
plt.show()

# Volume Total Unsigned Volume Vertical Current vs. Total Unsigned Vertical Current
print('Volume Total Unsigned Volume Vertical Current vs. Total Unsigned Vertical Current')
print('-' * 80)
print('Correlation:', df['Volume Total Unsigned Volume Vertical Current'].corr(df['Total Unsigned Vertical Current']))
print()
plt.scatter(df['Volume Total Unsigned Volume Vertical Current'], df['Total Unsigned Vertical Current'])
plt.xlabel('Volume Total Unsigned Volume Vertical Current')
plt.ylabel('Total Unsigned Vertical Current')
plt.title('Volume Total Unsigned Volume Vertical Current vs. Total Unsigned Vertical Current')
plt.show()

# Volume Total Unsigned Magnetic Flux vs. Phi
print('Volume Total Unsigned Magnetic Flux vs. Phi')
print('-' * 80)
print('Correlation:', df['Volume Total Unsigned Magnetic Flux'].corr(df['Phi']))
print()
plt.scatter(df['Volume Total Unsigned Magnetic Flux'], df['Phi'])
plt.xlabel('Volume Total Unsigned Magnetic Flux')
plt.ylabel('Phi')
plt.title('Volume Total Unsigned Magnetic Flux vs. Phi')
plt.show()

# Volume Mean Gradient of Total Magnetic Field vs. Gradient_10
print('Volume Mean Gradient of Total Magnetic Field vs. Gradient_10')
print('-' * 80)
print('Correlation:', df['Volume Mean Gradient of Total Magnetic Field'].corr(df['Gradient_10']))
print()
plt.scatter(df['Volume Mean Gradient of Total Magnetic Field'], df['Gradient_10'])
plt.xlabel('Volume Mean Gradient of Total Magnetic Field')
plt.ylabel('Gradient_10')
plt.title('Volume Mean Gradient of Total Magnetic Field vs. Gradient_10')
plt.show()
