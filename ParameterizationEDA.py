# Exploratory data analysis on the magnetic field volume parameterization CSV.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("InputData/volume_parameterizations.csv")

# Columns are ['Filename General', 'Total Magnetic Energy']

# Generate stats for the Total Magnetic Energy column and plot a histogram
print('Total Magnetic Energy stats:')
print(df['Total Magnetic Energy'].describe())

plt.hist(df['Total Magnetic Energy'], bins=30)
plt.title('Histogram of Total Magnetic Energy')
plt.xlabel('Total Magnetic Energy')
plt.ylabel('Frequency')
plt.show()


