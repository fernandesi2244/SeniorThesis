import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

file_path = '../OutputData/UnifiedActiveRegionData_5_15_150_100.csv'
df = pd.read_csv(file_path)

# print first 5 rows where 'Relevant Active Regions' is '[12192]'
print(df[df['Relevant Active Regions'] == '[12192]']['Filename General'].head())

# print first 5 rows where 'Relevant Active Regions' is '[12673]'
print(df[df['Relevant Active Regions'] == '[12673]']['Filename General'].head())

# print first 5 rows where 'Relevant Active Regions' is '[12158]'
print(df[df['Relevant Active Regions'] == '[12158]']['Filename General'].head())

exit()

# Define a function to plot with conditional coloring
def plot_with_conditional_coloring(ax, x, y, x_label, y_label, title):
    colors = np.where(df['Relevant Active Regions'] == '[12192]', 'red',
             np.where(df['Relevant Active Regions'] == '[12673]', 'green',
             np.where(df['Relevant Active Regions'] == '[12158]', 'orange', 'blue')))
    scatter = ax.scatter(x, y, c=colors)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    return scatter

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.subplots_adjust(hspace=0.7)#, wspace=0.4)

# Volume Total Unsigned Current Helicity vs. Total Unsigned Current Helicity
plot_with_conditional_coloring(axs[0, 0], df['Volume Total Unsigned Current Helicity'], df['Total Unsigned Current Helicity'],
                               'Volume Total Unsigned Current Helicity', 'Total Unsigned Current Helicity',
                               'Volume Total Unsigned Current Helicity vs. Total Unsigned Current Helicity')

# Volume Total Absolute Net Current Helicity vs. Abs of Net Current helicity
plot_with_conditional_coloring(axs[0, 1], df['Volume Total Absolute Net Current Helicity'], df['Abs of Net Current helicity'],
                               'Volume Total Absolute Net Current Helicity', 'Abs of Net Current helicity',
                               'Volume Total Absolute Net Current Helicity vs. Abs of Net Current helicity')

# Volume Total Unsigned Volume Vertical Current vs. Total Unsigned Vertical Current
plot_with_conditional_coloring(axs[1, 0], df['Volume Total Unsigned Volume Vertical Current'], df['Total Unsigned Vertical Current'],
                               'Volume Total Unsigned Volume Vertical Current', 'Total Unsigned Vertical Current',
                               'Volume Total Unsigned Volume Vertical Current vs. Total Unsigned Vertical Current')

# Volume Total Unsigned Magnetic Flux vs. Phi
plot_with_conditional_coloring(axs[1, 1], df['Volume Total Unsigned Magnetic Flux'], df['Phi'],
                               'Volume Total Unsigned Magnetic Flux', 'Phi',
                               'Volume Total Unsigned Magnetic Flux vs. Phi')

# Volume Mean Gradient of Total Magnetic Field vs. Gradient_10
plot_with_conditional_coloring(axs[2, 0], df['Volume Mean Gradient of Total Magnetic Field'], df['Gradient_10'],
                               'Volume Mean Gradient of Total Magnetic Field', 'Gradient_10',
                               'Volume Mean Gradient of Total Magnetic Field vs. Gradient_10')

# Hide the empty subplot (bottom right)
axs[2, 1].axis('off')

plt.show()
