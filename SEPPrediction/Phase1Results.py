import pandas as pd
import os

#~~~~

photospheric_grid_search_file = '../OutputData/sep_prediction_photospheric_data_grid_search_all_results_so_far.csv'
photospheric_grid_search_results = pd.read_csv(photospheric_grid_search_file)

print('Photospheric data shape before dropping dups:', photospheric_grid_search_results.shape)
photospheric_grid_search_results = photospheric_grid_search_results.drop_duplicates()
photospheric_grid_search_results = photospheric_grid_search_results.reset_index(drop=True)
print('Photospheric data shape after dropping dups:', photospheric_grid_search_results.shape)

#~~~~

coronal_grid_search_file = '../OutputData/sep_prediction_coronal_data_grid_search_all_results_so_far.csv'
coronal_grid_search_results = pd.read_csv(coronal_grid_search_file)

print('Coronal data shape before dropping dups:', coronal_grid_search_results.shape)
coronal_grid_search_results = coronal_grid_search_results.drop_duplicates()
coronal_grid_search_results = coronal_grid_search_results.reset_index(drop=True)
print('Coronal data shape after dropping dups:', coronal_grid_search_results.shape)

#~~~~

numeric_grid_search_file = '../OutputData/sep_prediction_numeric_data_grid_search_all_results_so_far.csv'
numeric_grid_search_results = pd.read_csv(numeric_grid_search_file)

print('Numeric data shape before dropping dups:', numeric_grid_search_results.shape)
numeric_grid_search_results = numeric_grid_search_results.drop_duplicates()
numeric_grid_search_results = numeric_grid_search_results.reset_index(drop=True)
print('Numeric data shape after dropping dups:', numeric_grid_search_results.shape)

#~~~~
"""
Print general performance stats for each file for each of the following columns:
- accuracy
- precision
- recall
- f1
- auc
"""
results_dfs = [photospheric_grid_search_results, coronal_grid_search_results, numeric_grid_search_results]
results_names = ['photospheric', 'coronal', 'numeric']
results_columns = ['accuracy', 'precision', 'recall', 'f1', 'auc']

for i, results_df in enumerate(results_dfs):
    print(f"Results for {results_names[i]} data:")
    for column in results_columns:
        mean_value = results_df[column].mean()
        std_value = results_df[column].std()
        min_value = results_df[column].min()
        max_value = results_df[column].max()
        percentile_25 = results_df[column].quantile(0.25)
        percentile_50 = results_df[column].quantile(0.5)
        percentile_75 = results_df[column].quantile(0.75)
        percentile_95 = results_df[column].quantile(0.95)

        print(f"  {column}: mean={mean_value:.4f}, std={std_value:.4f}, min={min_value:.4f}, max={max_value:.4f}, 25th={percentile_25:.4f}, 50th={percentile_50:.4f}, 75th={percentile_75:.4f}, 95th={percentile_95:.4f}")

    print()

"""
Now generate parallel coordinates plots for each df.
The score for the line colors is the f1 score.
Use the following columns for the axes:
- granularity (text column)
- oversampling_ratio (numeric column)
- n_features (numeric column)
- n_components (numeric column)
- model_type (text column)
"""
# Import statements
#...