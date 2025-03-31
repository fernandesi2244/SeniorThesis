import pandas as pd
import os

file_dir = '../OutputData/'
files_to_join = [
    'sep_prediction_photospheric_data_grid_search_all_results_so_far_pt1.csv',
    'sep_prediction_photospheric_data_grid_search_all_results_so_far_pt2.csv',
]

overall_results = pd.DataFrame()
for file_name in files_to_join:
    file_path = os.path.join(file_dir, file_name)
    if overall_results.empty:
        overall_results = pd.read_csv(file_path)
        print('Reading first file:', file_name, 'with shape:', overall_results.shape)
    else:
        new_data = pd.read_csv(file_path)
        overall_results = pd.concat([overall_results, new_data], ignore_index=True)
        print('Reading file:', file_name, 'with shape:', new_data.shape)
        print('Overall results shape:', overall_results.shape)

overall_results = overall_results.drop_duplicates()
overall_results = overall_results.reset_index(drop=True)

print('Final overall results shape after dropping dups:', overall_results.shape)

overall_results.to_csv(os.path.join(file_dir, 'sep_prediction_photospheric_data_grid_search_all_results_so_far.csv'), index=False)
