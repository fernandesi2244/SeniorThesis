import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np

# Load the grid search results
photospheric_grid_search_file = '../OutputData/sep_prediction_photospheric_data_grid_search_all_results_so_far.csv'
photospheric_grid_search_results = pd.read_csv(photospheric_grid_search_file)
print('Photospheric data shape before dropping dups:', photospheric_grid_search_results.shape)
photospheric_grid_search_results = photospheric_grid_search_results.drop_duplicates()
photospheric_grid_search_results = photospheric_grid_search_results.reset_index(drop=True)
print('Photospheric data shape after dropping dups:', photospheric_grid_search_results.shape)

coronal_grid_search_file = '../OutputData/sep_prediction_coronal_data_grid_search_all_results_so_far.csv'
coronal_grid_search_results = pd.read_csv(coronal_grid_search_file)
print('Coronal data shape before dropping dups:', coronal_grid_search_results.shape)
coronal_grid_search_results = coronal_grid_search_results.drop_duplicates()
coronal_grid_search_results = coronal_grid_search_results.reset_index(drop=True)
print('Coronal data shape after dropping dups:', coronal_grid_search_results.shape)

numeric_grid_search_file = '../OutputData/sep_prediction_numeric_data_grid_search_all_results_so_far.csv'
numeric_grid_search_results = pd.read_csv(numeric_grid_search_file)
print('Numeric data shape before dropping dups:', numeric_grid_search_results.shape)
numeric_grid_search_results = numeric_grid_search_results.drop_duplicates()
numeric_grid_search_results = numeric_grid_search_results.reset_index(drop=True)
print('Numeric data shape after dropping dups:', numeric_grid_search_results.shape)

# Calculate F-0.5 score for each dataframe
def calculate_f_half_score(precision, recall):
    """Calculate F-0.5 score which weights precision more than recall"""
    beta = 0.5
    beta_squared = beta ** 2
    if precision == 0 and recall == 0:
        return 0
    return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)

# Add F-0.5 score to each dataframe
for df in [photospheric_grid_search_results, coronal_grid_search_results, numeric_grid_search_results]:
    df['f_half'] = df.apply(lambda row: calculate_f_half_score(row['precision'], row['recall']), axis=1)

# Print general performance stats for each file
results_dfs = [photospheric_grid_search_results, coronal_grid_search_results, numeric_grid_search_results]
results_names = ['photospheric', 'coronal', 'numeric']
results_columns = ['accuracy', 'precision', 'recall', 'f1', 'f_half', 'auc']

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
        
        print(f" {column}: mean={mean_value:.4f}, std={std_value:.4f}, min={min_value:.4f}, max={max_value:.4f}, "
              f"25th={percentile_25:.4f}, 50th={percentile_50:.4f}, 75th={percentile_75:.4f}, 95th={percentile_95:.4f}")
    print()

# Generate parallel coordinates plots for each dataframe
# Columns to use: granularity, oversampling_ratio, n_features, n_components, model_type
parallel_coords_columns = ['granularity', 'oversampling_ratio', 'n_features', 'n_components', 'model_type']

def preprocess_for_parallel_coords(df):
    """
    Preprocess dataframe for parallel coordinates plot:
    - Convert categorical columns to numeric values
    - Create a copy to avoid modifying the original
    """
    df_copy = df.copy()
    
    # Convert granularity to numeric
    if 'granularity' in df_copy.columns:
        granularity_mapping = {val: i for i, val in enumerate(df_copy['granularity'].unique())}
        df_copy['granularity_num'] = df_copy['granularity'].map(granularity_mapping)
    
    # Convert model_type to numeric
    if 'model_type' in df_copy.columns:
        model_mapping = {val: i for i, val in enumerate(df_copy['model_type'].unique())}
        df_copy['model_type_num'] = df_copy['model_type'].map(model_mapping)
    
    # Sort df_copy in ascending order of 'f_half' instead of 'f1'
    df_copy = df_copy.sort_values(by='f_half', ascending=True)
    
    return df_copy, granularity_mapping, model_mapping

def create_parallel_coordinates_plot(df, data_type, output_dir='./'):
    """
    Create and save a parallel coordinates plot for the given dataframe
    """
    df_processed, granularity_mapping, model_mapping = preprocess_for_parallel_coords(df)
    
    # Reverse mappings for ticktext
    granularity_labels = {v: k for k, v in granularity_mapping.items()}
    model_labels = {v: k for k, v in model_mapping.items()}
    
    # Sort labels by their numeric values
    granularity_ticktext = [granularity_labels[i] for i in sorted(granularity_labels.keys())]
    model_ticktext = [model_labels[i] for i in sorted(model_labels.keys())]
    
    # Create dimensions list for parallel coordinates
    dimensions = [
        dict(
            range=[0, len(granularity_mapping)-1],
            tickvals=list(range(len(granularity_mapping))),
            ticktext=granularity_ticktext,
            label='Granularity',
            values=df_processed['granularity_num']
        ),
        dict(
            range=[df['oversampling_ratio'].min(), df['oversampling_ratio'].max()],
            label='Oversampling Ratio',
            values=df['oversampling_ratio']
        ),
        dict(
            range=[df['n_features'].min(), df['n_features'].max()],
            label='Number of Features',
            values=df['n_features']
        ),
        dict(
            range=[df['n_components'].min(), df['n_components'].max()],
            label='Number of Components',
            values=df['n_components']
        ),
        dict(
            range=[0, len(model_mapping)-1],
            tickvals=list(range(len(model_mapping))),
            ticktext=model_ticktext,
            label='Model Type',
            values=df_processed['model_type_num']
        )
    ]
    
    # Create figure - use f_half score for coloring instead of f1
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df['f_half'], colorscale='plasma', showscale=True, 
                     cmin=df['f_half'].min(), cmax=df['f_half'].max()),
            dimensions=dimensions,
        )
    )
    
    # Update layout - change title to reflect F-0.5 score
    fig.update_layout(
        title=f"Parallel Coordinates Plot for {data_type.capitalize()} Data (colored by F-0.5 Score)",
        font=dict(size=12),
        height=600,
        margin=dict(l=100, r=100, t=100, b=50)
    )
    
    # Save the figure
    output_file = os.path.join(output_dir, f'{data_type}_parallel_coordinates_f_half.html')
    fig.write_html(output_file)
    print(f"Saved parallel coordinates plot to {output_file}")
    
    return fig

# Create output directory if it doesn't exist
output_dir = '../OutputData/Phase1/Plots/New'
os.makedirs(output_dir, exist_ok=True)

# Generate and save parallel coordinates plots for each dataset
for i, (df, name) in enumerate(zip(results_dfs, results_names)):
    fig = create_parallel_coordinates_plot(df, name, output_dir)
    
# Show statistics about the best configurations based on F-0.5 score instead of F1 score
print("\nBest Configurations by F-0.5 Score:")
for i, (df, name) in enumerate(zip(results_dfs, results_names)):
    best_idx = df['f_half'].idxmax()
    best_config = df.iloc[best_idx]
    
    print(f"\nBest {name} configuration:")
    print(f"F-0.5 Score: {best_config['f_half']:.4f}")
    print(f"Accuracy: {best_config['accuracy']:.4f}")
    print(f"Precision: {best_config['precision']:.4f}")
    print(f"Recall: {best_config['recall']:.4f}")
    print(f"F1 Score: {best_config['f1']:.4f}")
    print(f"AUC: {best_config['auc']:.4f}")
    
    for col in parallel_coords_columns:
        print(f"{col}: {best_config[col]}")

# Print the harmonic mean and average of the F-0.5 scores for each model type for each dataset
print("\nHarmonic Mean and Average of F-0.5 Scores for Each Model Type:")
for i, (df, name) in enumerate(zip(results_dfs, results_names)):
    print(f"\n{name.capitalize()} Data:")
    model_types = df['model_type'].unique()
    
    for model in model_types:
        f_half_scores = df[df['model_type'] == model]['f_half']
        if len(f_half_scores) > 0:
            harmonic_mean_f_half = len(f_half_scores) / np.sum(1.0 / f_half_scores)
            average_f_half = f_half_scores.mean()
            print(f"Model Type: {model}, Harmonic Mean of F-0.5 Scores: {harmonic_mean_f_half:.4f}, Average of F-0.5 Scores: {average_f_half:.4f}")
        else:
            print(f"Model Type: {model}, No F-0.5 Scores available")
            continue

# Now do the same but condition on the granularity being per-disk-4hr
granularity = 'per-disk-4hr'
print(f"\nHarmonic Mean and Average of F-0.5 Scores for Each Model Type (Granularity: {granularity}):")
for i, (df, name) in enumerate(zip(results_dfs, results_names)):
    print(f"\n{name.capitalize()} Data:")
    model_types = df['model_type'].unique()
    
    for model in model_types:
        f_half_scores = df[(df['model_type'] == model) & (df['granularity'] == granularity)]['f_half']
        if len(f_half_scores) > 0:
            harmonic_mean_f_half = len(f_half_scores) / np.sum(1.0 / f_half_scores)
            average_f_half = f_half_scores.mean()
            print(f"Model Type: {model}, Harmonic Mean of F-0.5 Scores: {harmonic_mean_f_half:.4f}, Average of F-0.5 Scores: {average_f_half:.4f}")
        else:
            print(f"Model Type: {model}, No F-0.5 Scores available")
            continue