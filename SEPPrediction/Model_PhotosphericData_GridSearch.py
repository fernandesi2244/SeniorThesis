from PhotosphericDataLoader import SEPInputDataGenerator
from ModelConstructor import ModelConstructor
import pandas as pd
import numpy as np
import time
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import random
from sklearn.utils import shuffle

NAME = 'sep_prediction_photospheric_data_grid_search'

# Create output directory for results
RESULTS_DIR = f'results/photospheric_data'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Multiprocessing setup
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

def build_feature_names(granularity):
    """
    Build a list of feature names based on the SEPInputDataGenerator's column definitions.
    
    Returns:
        List of feature names
    """
    if granularity == 'per-blob':
        # One-time features
        feature_names = list(SEPInputDataGenerator.BLOB_ONE_TIME_INFO)
        
        # Time-series features
        for t in range(SEPInputDataGenerator.TIMESERIES_STEPS):
            for col in SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL:
                if t == 0:
                    feature_names.append(f"{col}")
                else:
                    feature_names.append(f"{col}_t-{t*4}")
    elif granularity.startswith('per-disk'):
        """
        In the per-disk setting, there are all of the above features but for the top
        5 blobs of each disk at that time. This means that the feature names will be
        repeated for each blob, but with a suffix to indicate the blob number.
        """

        # Process one-time features and time-series features for each blob

        # One-time info for disk
        feature_names = list(SEPInputDataGenerator.BLOB_ONE_TIME_INFO)

        # Time-series features for top 5 disk blobs and their previous 5 time steps
        for i in range(1, 6):
            for t in range(SEPInputDataGenerator.TIMESERIES_STEPS):
                for col in SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL:
                    if t == 0:
                        feature_names.append(f"{col}_blob{i}")
                    else:
                        if granularity == 'per-disk-4hr':
                            feature_names.append(f"{col}_t-{t*4}_blob{i}")
                        elif granularity == 'per-disk-1d':
                            feature_names.append(f"{col}_t-{t*24}_blob{i}")
    else:
        raise ValueError(f"Invalid granularity: {granularity}")
    
    return feature_names

def extract_all_data(generator):
    """
    Extract all data from a generator
    
    Args:
        generator: SEPInputDataGenerator instance
        
    Returns:
        X: Features array
        y: Labels array
    """
    all_X = []
    all_y = []
    for i in range(len(generator)):
        if i % 100 == 0:
            print(f'Extracting batch {i+1} of {len(generator)}...')
        X_batch, y_batch = generator[i]
        all_X.append(X_batch)
        all_y.append(y_batch)
    
    if len(all_X) > 0:
        return np.vstack(all_X), np.concatenate(all_y)
    else:
        return np.array([]), np.array([])

def evaluate_model(y_true, y_pred, y_pred_proba, set_name=""):
    """
    Evaluate model performance with various metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities for positive class
        set_name: Name of the dataset being evaluated (e.g., "Validation", "Test")
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print(f'{set_name} Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Confusion Matrix:')
    print(cm)
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def feature_selection(X_train, y_train, feature_names):
    """
    Perform feature selection using Random Forest
    
    Args:
        X_train: Training data
        y_train: Training labels
        feature_names: List of feature names
        
    Returns:
        Indices of features sorted by importance
    """
    print('\nPerforming feature selection...')
    
    # Initialize Random Forest for feature selection
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=cpus_to_use
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    
    # Create DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'StdDev': std
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    feature_indices = [feature_names.index(feature) for feature in feature_importance_df['Feature']]
    # Get indices of features sorted by importance
    
    # Plot top 30 features
    plt.figure(figsize=(12, 10))
    top_n = 30
    plt.barh(feature_importance_df['Feature'][:top_n], 
            feature_importance_df['Importance'][:top_n], 
            xerr=feature_importance_df['StdDev'][:top_n],
            color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features by Importance')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{NAME}_top_{top_n}_features.png', dpi=300)
    plt.close()
    
    # Save feature importance
    feature_importance_df.to_csv(f'{RESULTS_DIR}/{NAME}_feature_importance.csv', index=False)
    
    # Return indices of features sorted by importance
    return feature_indices

def evaluate_pca_component(X_train, y_train, X_val, y_val, n_features, n_components):
    """
    Evaluate a specific PCA component count for a given feature set
    
    Args:
        X_train: Training data with selected features
        y_train: Training labels
        X_val: Validation data with selected features
        y_val: Validation labels
        n_features: Number of features used
        n_components: Number of PCA components to test
        
    Returns:
        Dictionary with evaluation results
    """
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    # Calculate variance explained
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    
    # Train Random Forest on transformed data
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=cpus_to_use
    )
    
    rf.fit(X_train_pca, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_val_pca)
    y_pred_proba = rf.predict_proba(X_val_pca)[:, 1]
    
    # Calculate metrics
    metrics = evaluate_model(y_val, y_pred, y_pred_proba)
    
    # Return results
    return {
        'n_features': n_features,
        'n_components': n_components,
        'explained_variance': explained_variance,
        'metrics': metrics,
        'pca': pca,
        'model': rf
    }

def plot_results(results_df, metric='f1'):
    """
    Plot results for different feature counts and PCA components
    
    Args:
        results_df: DataFrame with results
        metric: Metric to plot (default: f1)
    """
    pivot_data = results_df.groupby(['n_components', 'n_features'])[metric].mean().reset_index()
    
    # Create pivot table for heatmap
    pivot_df = pivot_data.pivot(index='n_components', columns='n_features', values=metric)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.4f')
    plt.title(f'{metric.upper()} Score by Feature Count and PCA Components')
    plt.xlabel('Number of Features')
    plt.ylabel('Number of PCA Components')
    plt.savefig(f'{RESULTS_DIR}/{NAME}_{metric}_heatmap.png', dpi=300)
    plt.close()
    
    # For each feature count, plot metrics vs number of components
    plt.figure(figsize=(15, 10))
    feature_counts = results_df['n_features'].unique()
    
    for feat_count in feature_counts:
        # Use the grouped data to plot the lines
        subset = pivot_data[pivot_data['n_features'] == feat_count]
        plt.plot(subset['n_components'], subset[metric], marker='o', label=f'{feat_count} features')
    
    plt.title(f'{metric.upper()} Score vs PCA Components')
    plt.xlabel('Number of PCA Components')
    plt.ylabel(f'{metric.upper()} Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{RESULTS_DIR}/{NAME}_{metric}_by_components.png', dpi=300)
    plt.close()

def load_data(granularity):
    if granularity == 'per-blob':
        train_features_file = f'{NAME}_X_train_data_per_blob.npy'
        train_labels_file = f'{NAME}_y_train_data_per_blob.npy'
        val_features_file = f'{NAME}_val_data_per_blob.npy'
        val_labels_file = f'{NAME}_val_labels_per_blob.npy'
        test_features_file = f'{NAME}_test_data_per_blob.npy'
        test_labels_file = f'{NAME}_test_labels_per_blob.npy'
    elif granularity == 'per-disk-4hr':
        train_features_file = f'{NAME}_X_train_data_per_disk_4hr.npy'
        train_labels_file = f'{NAME}_y_train_data_per_disk_4hr.npy'
        val_features_file = f'{NAME}_val_data_per_disk_4hr.npy'
        val_labels_file = f'{NAME}_val_labels_per_disk_4hr.npy'
        test_features_file = f'{NAME}_test_data_per_disk_4hr.npy'
        test_labels_file = f'{NAME}_test_labels_per_disk_4hr.npy'
    elif granularity == 'per-disk-1d':
        train_features_file = f'{NAME}_X_train_data_per_disk_1d.npy'
        train_labels_file = f'{NAME}_y_train_data_per_disk_1d.npy'
        val_features_file = f'{NAME}_val_data_per_disk_1d.npy'
        val_labels_file = f'{NAME}_val_labels_per_disk_1d.npy'
        test_features_file = f'{NAME}_test_data_per_disk_1d.npy'
        test_labels_file = f'{NAME}_test_labels_per_disk_1d.npy'
    else:
        raise ValueError(f"Invalid granularity: {granularity}")
    
    if os.path.exists(test_labels_file):
        # assuming all files exist if the last-generated one does
        X_train = np.load(train_features_file)
        y_train = np.load(train_labels_file)
        X_val = np.load(val_features_file)
        y_val = np.load(val_labels_file)
        X_test = np.load(test_features_file)
        y_test = np.load(test_labels_file)
    else:
        # Load dataset
        blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv'
        blob_df = pd.read_csv(blob_df_filename)
        
        print(f"Loaded dataset with {len(blob_df)} rows")
        
        # Preprocess the data
        blob_df['Produced an SEP'] = (blob_df['Number of SEPs Produced'] > 0) * 1  # 1 if produced, 0 otherwise
        blob_df['Year'] = blob_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
        blob_df['Is Plage'] = blob_df['Is Plage'].astype(int)
        blob_df['Most Probable AR Num'] = blob_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
        
        # Split the data by year to maintain the same structure as in the original scripts
        years = blob_df['Year'].unique()
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        for year in years:
            blobs_in_year = blob_df[blob_df['Year'] == year]
            print(f'Year: {year}, Number of blobs: {len(blobs_in_year)}')
            print(f'Number of SEPs: {blobs_in_year["Produced an SEP"].sum()}')
        
            # Group by active region to prevent data leakage
            grouped = blobs_in_year.groupby('Most Probable AR Num')['Produced an SEP'].max()
        
            # Handle special case with only one SEP-producing active region
            if grouped[grouped == 1].count() == 1:
                min_train_regions = grouped[grouped == 1].index
                remaining_train_regions, test_regions = train_test_split(
                    grouped[grouped == 0].index, test_size=0.2, random_state=42
                )
                train_regions = np.concatenate([min_train_regions, remaining_train_regions])
            else:
                train_regions, test_regions = train_test_split(
                    grouped.index, test_size=0.2, stratify=grouped, random_state=42
                )
        
            # Select records based on their active region
            train_from_year = blobs_in_year[blobs_in_year['Most Probable AR Num'].isin(train_regions)]
            test_from_year = blobs_in_year[blobs_in_year['Most Probable AR Num'].isin(test_regions)]
        
            # Further split train into train and validation
            grouped_train = train_from_year.groupby('Most Probable AR Num')['Produced an SEP'].max()
            
            if grouped_train[grouped_train == 1].count() == 1:
                min_train_regions = grouped_train[grouped_train == 1].index
                remaining_train_regions, val_regions = train_test_split(
                    grouped_train[grouped_train == 0].index, test_size=0.25, random_state=42
                )
                train_regions = np.concatenate([min_train_regions, remaining_train_regions])
            else:
                train_regions, val_regions = train_test_split(
                    grouped_train.index, test_size=0.25, stratify=grouped_train, random_state=42
                )
        
            new_train_from_year = train_from_year[train_from_year['Most Probable AR Num'].isin(train_regions)]
            val_from_year = train_from_year[train_from_year['Most Probable AR Num'].isin(val_regions)]
        
            # Print statistics
            print(f'Train set: {len(new_train_from_year)}, Val set: {len(val_from_year)}, Test set: {len(test_from_year)}')
            print(f'Train SEPs: {new_train_from_year["Produced an SEP"].sum()}, Val SEPs: {val_from_year["Produced an SEP"].sum()}, Test SEPs: {test_from_year["Produced an SEP"].sum()}')
        
            if len(new_train_from_year) > 0 and len(val_from_year) > 0 and len(test_from_year) > 0:
                train_df = pd.concat([train_df, new_train_from_year])
                val_df = pd.concat([val_df, val_from_year])
                test_df = pd.concat([test_df, test_from_year])
            else:
                print(f'Year {year} has insufficient data in one of the sets. Skipping.')
        
        # Create the data generators (right now, just being used to get timeseries data)
        print('\nCreating data generators...')
        batch_size = 64
        train_generator = SEPInputDataGenerator(train_df, batch_size, False, granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
        val_generator = SEPInputDataGenerator(val_df, batch_size, False, granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
        test_generator = SEPInputDataGenerator(test_df, batch_size, False, granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
        
        # Extract all data from generators
        print('\nExtracting data from generators...')
        X_train, y_train = extract_all_data(train_generator)
        X_val, y_val = extract_all_data(val_generator)
        X_test, y_test = extract_all_data(test_generator)

        # Save the data
        np.save(train_features_file, X_train)
        np.save(train_labels_file, y_train)

        np.save(val_features_file, X_val)
        np.save(val_labels_file, y_val)

        np.save(test_features_file, X_test)
        np.save(test_labels_file, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_model(model_file):
    """
    Load a model from a file
    
    Args:
        model_file: File path to the model
    
    Returns:
        Model object
    """
    return joblib.load(model_file)

def main():
    """Main function to run the combined feature selection and PCA analysis"""
    start_time = time.time()
    print(f"Starting combined feature selection and PCA analysis at {time.ctime()}")

    #granularities = ['per-blob', 'per-disk-4hr', 'per-disk-1d'] # ['per-blob', 'per-disk-4hr', 'per-disk-1d']
    granularities = ['per-disk-1d', 'per-disk-4hr', 'per-blob']

    oversampling_ratios = [0.1, 0.25, 0.5, 0.65, 0.75, 1] # [0.1, 0.25, 0.5, 0.65, 0.75, 1] # pos:neg ratio. TODO: figure out some other day why > 0.65 isn't working
    
    # Define feature counts to test
    feature_counts = [-1, 20, 40, 60, 80, 100] #[20, 40, 60, 80, 100]
    
    # Define component counts to test for PCA
    component_counts = [-1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50] #[-1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

    # Model files
    # model_types = [
    #     'random_forest_simple',
    #     'random_forest_complex',
    #     'isolation_forest',
    #     #'gaussian_RBF',
    #     #'gaussian_matern',
    #     'nn_simple',
    #     'nn_complex',
    #     'logistic_regression_v1',
    #     'logistic_regression_v2',
    #     'gbm',
    #     'lightgbm',
    #     'xgboost',
    #     'svm_rbf',
    #     'svm_poly',
    #     'knn_v1',
    #     'knn_v2',
    #     'knn_v3',
    # ]

    model_types = [
        'random_forest_simple',
        'random_forest_complex',
        'isolation_forest',
        #'gaussian_RBF',
        #'gaussian_matern',
        'nn_simple',
        'nn_complex',
        'logistic_regression_v1',
        'logistic_regression_v2',
        'gbm',
        # 'lightgbm',
        'xgboost',
        'svm_rbf',
        'svm_poly',
        'knn_v1',
        'knn_v2',
        'knn_v3',
    ]

    # TODO: later, look at ensembling techniques
    
    # Create a list to store all results
    all_results = []
    # best_f1 = 0
    # best_config = None
    # best_model = None
    # best_pca = None

    for model_type in model_types:
        print('\n' + '-'*50)
        print(f'\nEvaluating model: {model_type}')
        print('-'*50)

        for granularity in granularities:
            print('\n' + '-'*50)
            print(f'\nEvaluating granularity: {granularity}')
            print('-'*50)
            # Build feature names for interpretation
            feature_names = build_feature_names(granularity)

            # Load the data
            X_train_OG, y_train_OG, X_val, y_val, X_test, y_test = load_data(granularity)

            # Check for NaN and Inf values
            print('\nChecking for NaN and Inf values:')
            print(f'Train NaN count: {np.isnan(X_train_OG).sum()}, Inf count: {np.isinf(X_train_OG).sum()}')
            print(f'Val NaN count: {np.isnan(X_val).sum()}, Inf count: {np.isinf(X_val).sum()}')
            print(f'Test NaN count: {np.isnan(X_test).sum()}, Inf count: {np.isinf(X_test).sum()}')
            
            # Replace any NaN or Inf values with 0. NOTE: This code should never change the data since there are no NaNs.
            X_train_OG = np.nan_to_num(X_train_OG, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Standardize the features
            scaler = StandardScaler()
            X_train_OG = scaler.fit_transform(X_train_OG)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            for oversampling_ratio in oversampling_ratios:
                print('\n' + '-'*50)
                print(f'\nEvaluating oversampling ratio: {oversampling_ratio}')
                print('-'*50)
            
                # Apply class balancing to training set
                print('\nBefore resampling:')
                print('Train set count:', len(X_train_OG))
                print('Train set SEP count:', np.sum(y_train_OG))

                # if oversampling ratio is less than or equal to the current ratio, skip
                if oversampling_ratio <= np.sum(y_train_OG) / len(y_train_OG):
                    print(f"Skipping oversampling ratio {oversampling_ratio} as positive class already significant enough.")
                    continue
                
                # Over-sample minority class
                ros = RandomOverSampler(sampling_strategy=oversampling_ratio/2, random_state=42)
                X_train, y_train = ros.fit_resample(X_train_OG, y_train_OG)
                
                # Under-sample majority class
                rus = RandomUnderSampler(sampling_strategy=oversampling_ratio, random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)

                # Reshuffle the data
                X_train, y_train = shuffle(X_train, y_train, random_state=42)
                
                print('After resampling:')
                print('Train set count:', len(X_train))
                print('Train set SEP count:', np.sum(y_train))
                
                # Print dataset statistics
                print('\nDataset Statistics:')
                print(f'Train set size: {len(X_train)}, SEP events: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)')
                print(f'Validation set size: {len(X_val)}, SEP events: {np.sum(y_val)} ({np.mean(y_val)*100:.2f}%)')
                print(f'Test set size: {len(X_test)}, SEP events: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)')
                print(f'Number of features: {X_train.shape[1]}')

                # Run feature selection
                feature_indices = feature_selection(X_train, y_train, feature_names)
                features = [feature_names[i] for i in feature_indices]
                
                # Loop through different feature counts
                for n_features in feature_counts:
                    if n_features == -1:
                        print('Skipping feature selection...')
                        n_features = len(feature_indices)

                    if n_features > len(feature_indices):
                        print(f"Warning: Requested {n_features} features, but only {len(feature_indices)} available. Using all available features.")
                        n_features = len(feature_indices)
                    
                    # Get the top n features
                    selected_indices = feature_indices[:n_features]
                    
                    # Extract data with selected features
                    X_train_selected = X_train[:, selected_indices]
                    X_val_selected = X_val[:, selected_indices]
                    X_test_selected = X_test[:, selected_indices]
                    
                    print(f"\nEvaluating top {n_features} features...")
                    
                    # Define valid component counts based on number of features
                    valid_components = [c for c in component_counts if c <= min(n_features, X_train_selected.shape[0])]
                    
                    # Loop through different PCA component counts
                    for n_components in valid_components:
                        if granularity == 'per-blob' and n_components == -1:
                            print('Skipping non-PCA analysis for per-blob granularity...')
                            continue

                        # now, if n_components == -1, then it must be that granularity is full-disk
                        if n_components == -1 and model_type.startswith('nn') and not n_features == -1:
                            print('Skipping non-PCA analysis for full-disk NNs where feature reduction occurs')
                            continue

                        print('\n' + '-'*50)
                        print(f'Feature Count: {n_features}, PCA Components: {n_components}')
                        if n_components == -1:
                            print('Skipping PCA...')
                        print('-'*50)

                        # time model loading
                        if model_type == 'isolation_forest':
                            percent_pos = np.sum(y_train) / len(y_train)
                            model = ModelConstructor.create_model('photospheric', model_type, granularity, n_components, contamination=percent_pos)
                        else:
                            model = ModelConstructor.create_model('photospheric', model_type, granularity, n_components)

                        if n_components != -1:
                            # Apply PCA
                            pca = PCA(n_components=n_components, random_state=42)
                            X_train_pca = pca.fit_transform(X_train_selected)
                            X_val_pca = pca.transform(X_val_selected)
                        
                            # Calculate variance explained
                            explained_variance = np.sum(pca.explained_variance_ratio_) * 100
                        else:
                            pca = None
                            X_train_pca = X_train_selected
                            X_val_pca = X_val_selected
                            explained_variance = 100
                        
                        # time model training
                        train_start = time.time()
                        model.fit(X_train_pca, y_train)
                        train_end = time.time()
                        print(f'Model {model_type} trained in {train_end - train_start:.2f} seconds')
                        
                        # Make predictions
                        if model_type == 'isolation_forest':
                            # y_pred is 1 for inliers, -1 for outliers, but we want
                            # 1 for outliers and 0 for inliers
                            y_pred = model.predict(X_val_pca)
                            y_pred[y_pred == 1] = 0
                            y_pred[y_pred == -1] = 1
                        elif model_type == 'nn_simple' or model_type == 'nn_complex':
                            y_pred_proba = model.predict(X_val_pca)
                            # Assume 0.5 threshold for now, prob need to optimize over too
                            y_pred = (y_pred_proba > 0.5).astype(int)
                        else:
                            y_pred = model.predict(X_val_pca)

                        if model_type == 'isolation_forest':
                            anomaly_scores = model.decision_function(X_val_pca)
                            y_pred_proba = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
                        elif model_type == 'nn_simple' or model_type == 'nn_complex':
                            pass # already calculated
                        else:
                            y_pred_proba = model.predict_proba(X_val_pca)[:, 1]
                        
                        # Calculate metrics
                        metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Validation")
                        
                        # Extract metrics to store in results list
                        # TODO: Update this and later with all selected hyperparameters
                        result_row = {
                            'granularity': granularity,
                            'scaler': scaler,
                            'oversampling_ratio': oversampling_ratio,
                            'n_features': n_features,
                            'feature_indices': selected_indices,
                            'feature_names': features,
                            'n_components': n_components,
                            'pca': pca,
                            'model_type': model_type,
                            'model': model,
                            'explained_variance': explained_variance,
                            'accuracy': metrics['accuracy'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'auc': metrics['auc']
                        }
                        
                        all_results.append(result_row)

                        # Save current state of all_results
                        results_df = pd.DataFrame(all_results)
                        results_df.to_csv(f'{RESULTS_DIR}/{NAME}_all_results_so_far.csv', index=False)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(f'{RESULTS_DIR}/{NAME}_all_results.csv', index=False)
    
    # Plot results
    plot_results(results_df, 'f1')
    plot_results(results_df, 'accuracy')
    plot_results(results_df, 'precision')
    plot_results(results_df, 'recall')
    plot_results(results_df, 'auc')

    # Get best configuration
    best_config = results_df.loc[results_df['f1'].idxmax()].to_dict()
    
    # Print best configuration
    print("\nBest Configuration:")
    print(f"Model type: {best_config['model_type']}")
    print(f"Granularity: {best_config['granularity']}")
    print(f"Oversampling Ratio: {best_config['oversampling_ratio']}")
    print(f"Number of Features: {best_config['n_features']}")
    print(f"Best features: {best_config['feature_names']}")
    print(f"Number of PCA Components: {best_config['n_components']}")
    print(f"F1 Score: {best_config['f1']:.4f}")
    print(f"Accuracy: {best_config['accuracy']:.4f}")
    print(f"Precision: {best_config['precision']:.4f}")
    print(f"Recall: {best_config['recall']:.4f}")
    print(f"AUC: {best_config['auc']:.4f}")
    
    # Save best feature indices
    best_feature_names = best_config['feature_names']
    pd.DataFrame({'Feature': best_feature_names}).to_csv(f'{RESULTS_DIR}/{NAME}_best_features.csv', index=False)
    
    # Now, evaluate the best model on the test set
    print("\nEvaluating best model on test set...")
    
    # Get the best configuration's selected features
    _, _, _, _, X_test, y_test = load_data(best_config['granularity'])
    X_test = best_config['scaler'].transform(X_test)
    X_test_selected = X_test[:, best_config['feature_indices']]
    
    # Apply PCA transformation
    if best_config['n_components'] == -1:
        X_test_pca = X_test_selected
    else:
        X_test_pca = best_config['pca'].transform(X_test_selected)
    
    # Make predictions
    if best_config['model_type'] == 'isolation_forest':
        y_test_pred = best_config['model'].predict(X_test_pca)
        y_test_pred[y_test_pred == 1] = 0
        y_test_pred[y_test_pred == -1] = 1

        anomaly_scores = best_config['model'].decision_function(X_test_pca)
        y_pred_proba = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    elif best_config['model_type'] == 'nn_simple' or best_config['model_type'] == 'nn_complex':
        y_test_pred_proba = best_config['model'].predict(X_test_pca)
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    else:
        y_test_pred = best_config['model'].predict(X_test_pca)
        y_test_pred_proba = best_config['model'].predict_proba(X_test_pca)[:, 1]
    
    # Evaluate performance
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Save the best model, PCA transformer, and feature indices
    model_metadata = {
        'model_type': best_config['model_type'],
        'granularity': best_config['granularity'],
        'oversampling_ratio': best_config['oversampling_ratio'],
        'feature_indices': best_config['feature_indices'],
        'feature_names': best_feature_names,
        'n_components': best_config['n_components'],
        'train_metrics': {
            'accuracy': best_config['accuracy'],
            'precision': best_config['precision'],
            'recall': best_config['recall'],
            'f1': best_config['f1'],
            'auc': best_config['auc']
        },
        'test_metrics': test_metrics,
    }
    
    # Save model and metadata
    joblib.dump(best_config['model'], f'{RESULTS_DIR}/{NAME}_best_model.joblib')
    joblib.dump(best_config['pca'], f'{RESULTS_DIR}/{NAME}_best_pca.joblib')
    joblib.dump(best_config['scaler'], f'{RESULTS_DIR}/{NAME}_best_scaler.job')
    joblib.dump(model_metadata, f'{RESULTS_DIR}/{NAME}_metadata.joblib')
    
    print(f"\nBest model and metadata saved to {RESULTS_DIR}/ directory")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nAnalysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')

if __name__ == "__main__":
    main()
