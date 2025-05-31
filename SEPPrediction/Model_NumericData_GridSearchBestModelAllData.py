from NumericDataLoader import SEPInputDataGenerator
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
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import random
from sklearn.utils import shuffle

# leave this the same for access to datasets from 2nd round of grid search
NAME = 'sep_prediction_numeric_data_grid_search2'

# Create output directory for results
RESULTS_DIR = f'results/numeric_data'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use only split 42
SPLIT_SEED = 42

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
        for i in range(1, SEPInputDataGenerator.TOP_N_BLOBS + 1):
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

    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Calculate HSS and TSS
    hss = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (TN + FP))
    tss = TP / (TP + FN) - FP / (FP + TN)
    
    # Print metrics
    print(f'{set_name} Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'HSS: {hss:.4f}')
    print(f'TSS: {tss:.4f}')
    print(f'Confusion Matrix:')
    print(cm)
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'hss': hss,
        'tss': tss,
        'confusion_matrix': cm
    }

def evaluate_from_combined_confusion_matrix(combined_cm, set_name=""):
    """
    Calculate metrics from a combined confusion matrix
    
    Args:
        combined_cm: The combined confusion matrix from multiple splits
        set_name: Name of the dataset being evaluated
        
    Returns:
        Dictionary of metrics
    """
    TN, FP, FN, TP = combined_cm.ravel()
    
    # Calculate metrics directly from confusion matrix
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate HSS and TSS
    hss = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (TN + FP))
    tss = TP / (TP + FN) - FP / (FP + TN) if (TP + FN) > 0 and (FP + TN) > 0 else 0
    
    # For AUC, we can't easily compute it from just a confusion matrix without the probabilities
    # We'll set it to None for now
    auc = None
    
    def get_f_half(precision, recall):
        return (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
    
    # Print metrics
    print(f'{set_name} Results from Combined Confusion Matrix:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'F-0.5 Score: {get_f_half(precision, recall):.4f}')
    print(f'HSS: {hss:.4f}')
    print(f'TSS: {tss:.4f}')
    print(f'Combined Confusion Matrix:')
    print(combined_cm)
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,  # Will be None
        'hss': hss,
        'tss': tss,
        'confusion_matrix': combined_cm
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
    
    # Create DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    feature_indices = [feature_names.index(feature) for feature in feature_importance_df['Feature']]
    
    # Save feature importance
    feature_importance_df.to_csv(f'{RESULTS_DIR}/{NAME}_feature_importance_alldata.csv', index=False)
    
    # Return indices of features sorted by importance
    return feature_indices

def load_data(granularity):
    """
    Load data for a specific granularity, combining ALL data for training
    
    Args:
        granularity: Data granularity ('per-blob', 'per-disk-4hr', or 'per-disk-1d')
        
    Returns:
        X_all, y_all, X_test, y_test: Data arrays (X_all = train + val + test, X_test for metrics)
    """
    if granularity == 'per-blob':
        train_features_file = f'{NAME}_X_train_data_per_blob_split{SPLIT_SEED}.npy'
        train_labels_file = f'{NAME}_y_train_data_per_blob_split{SPLIT_SEED}.npy'
        val_features_file = f'{NAME}_val_data_per_blob_split{SPLIT_SEED}.npy'
        val_labels_file = f'{NAME}_val_labels_per_blob_split{SPLIT_SEED}.npy'
        test_features_file = f'{NAME}_test_data_per_blob_split{SPLIT_SEED}.npy'
        test_labels_file = f'{NAME}_test_labels_per_blob_split{SPLIT_SEED}.npy'
    elif granularity == 'per-disk-4hr':
        train_features_file = f'{NAME}_X_train_data_per_disk_4hr_split{SPLIT_SEED}.npy'
        train_labels_file = f'{NAME}_y_train_data_per_disk_4hr_split{SPLIT_SEED}.npy'
        val_features_file = f'{NAME}_val_data_per_disk_4hr_split{SPLIT_SEED}.npy'
        val_labels_file = f'{NAME}_val_labels_per_disk_4hr_split{SPLIT_SEED}.npy'
        test_features_file = f'{NAME}_test_data_per_disk_4hr_split{SPLIT_SEED}.npy'
        test_labels_file = f'{NAME}_test_labels_per_disk_4hr_split{SPLIT_SEED}.npy'
    elif granularity == 'per-disk-1d':
        train_features_file = f'{NAME}_X_train_data_per_disk_1d_split{SPLIT_SEED}.npy'
        train_labels_file = f'{NAME}_y_train_data_per_disk_1d_split{SPLIT_SEED}.npy'
        val_features_file = f'{NAME}_val_data_per_disk_1d_split{SPLIT_SEED}.npy'
        val_labels_file = f'{NAME}_val_labels_per_disk_1d_split{SPLIT_SEED}.npy'
        test_features_file = f'{NAME}_test_data_per_disk_1d_split{SPLIT_SEED}.npy'
        test_labels_file = f'{NAME}_test_labels_per_disk_1d_split{SPLIT_SEED}.npy'
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
        
        # Combine ALL sets (train + validation + test) for training on all available data
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        
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
                    grouped_train[grouped_train == 0].index, test_size=0.25, random_state=SPLIT_SEED
                )
                train_regions = np.concatenate([min_train_regions, remaining_train_regions])
            else:
                train_regions, val_regions = train_test_split(
                    grouped_train.index, test_size=0.25, stratify=grouped_train, random_state=SPLIT_SEED
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

        # Save the data with split_seed in filename
        np.save(train_features_file, X_train)
        np.save(train_labels_file, y_train)
        np.save(val_features_file, X_val)
        np.save(val_labels_file, y_val)
        np.save(test_features_file, X_test)
        np.save(test_labels_file, y_test)

        # Combine ALL sets (train + validation + test) for training on all available data
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

    return X_all, y_all, X_test, y_test

def main():
    """Main function to run the grid search training on all data"""
    start_time = time.time()
    print(f"Starting numeric data training on all available data at {time.ctime()}")

    granularities = ['per-disk-4hr']
    oversampling_ratios = [0.5]
    feature_counts = [60]
    component_counts = [-1]
    
    model_types = [
        'random_forest_complex',
    ]

    # Create a list to store all results
    all_results = []

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

            for oversampling_ratio in oversampling_ratios:
                print('\n' + '-'*50)
                print(f'\nEvaluating oversampling ratio: {oversampling_ratio}')
                print('-'*50)
                
                # For each feature count
                for n_features in feature_counts:
                    print('\n' + '-'*50)
                    print(f'\nEvaluating feature count: {n_features}')
                    print('-'*50)
                    
                    # For each PCA component count
                    for n_components in component_counts:
                        print('\n' + '-'*50)
                        print(f'\nEvaluating component count: {n_components}')
                        print('-'*50)

                        if n_features != -1 and n_components > n_features:
                            print(f"Skipping PCA with {n_components} components as it exceeds the number of features {n_features}.")
                            continue
                            
                        print('\n' + '-'*50)
                        print(f'\nTraining final model on all available data')
                        print('-'*50)
                        
                        # Define model name
                        model_name = f"{NAME}_{granularity}_{oversampling_ratio}_{model_type}_{n_features}_{n_components}_alldata"
                        
                        # Load the data (X_all = train + validation + test combined)
                        X_all_OG, y_all_OG, X_test, y_test = load_data(granularity)

                        # Check for NaN and Inf values
                        print('\nChecking for NaN and Inf values:')
                        print(f'All data NaN count: {np.isnan(X_all_OG).sum()}, Inf count: {np.isinf(X_all_OG).sum()}')
                        print(f'Test NaN count: {np.isnan(X_test).sum()}, Inf count: {np.isinf(X_test).sum()}')
                        
                        # Replace any NaN or Inf values with 0
                        X_all_OG = np.nan_to_num(X_all_OG, nan=0.0, posinf=0.0, neginf=0.0)
                        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

                        # Standardize the features
                        scaler = StandardScaler()
                        X_all_OG = scaler.fit_transform(X_all_OG)
                        X_test = scaler.transform(X_test)
                        
                        # Apply resampling if needed
                        print('\nBefore resampling:')
                        print('All data set count:', len(X_all_OG))
                        print('All data set SEP count:', np.sum(y_all_OG))
                        print('Test set count (for metrics only):', len(X_test))
                        print('Test set SEP count (for metrics only):', np.sum(y_test))

                        # if oversampling ratio is less than or equal to the current ratio, skip this configuration
                        if oversampling_ratio <= np.sum(y_all_OG) / (len(y_all_OG) - np.sum(y_all_OG)):
                            print(f"Skipping oversampling ratio {oversampling_ratio} as positive class already significant enough.")
                            continue
                        
                        # Over-sample minority class
                        ros = RandomOverSampler(sampling_strategy=oversampling_ratio/2, random_state=SPLIT_SEED)
                        X_all, y_all = ros.fit_resample(X_all_OG, y_all_OG)
                        
                        # Under-sample majority class
                        rus = RandomUnderSampler(sampling_strategy=oversampling_ratio, random_state=SPLIT_SEED)
                        X_all, y_all = rus.fit_resample(X_all, y_all)

                        # Reshuffle the data
                        X_all, y_all = shuffle(X_all, y_all, random_state=SPLIT_SEED)
                        
                        print('After resampling:')
                        print('All data set count:', len(X_all))
                        print('All data set SEP count:', np.sum(y_all))
                        
                        # Run feature selection
                        feature_indices = feature_selection(X_all, y_all, feature_names)
                        selected_features = [feature_names[i] for i in feature_indices]
                        
                        # Check if we have enough features
                        if n_features == -1:
                            print('Using all features...')
                            n_features = len(feature_indices)

                        if n_features > len(feature_indices):
                            if -1 in feature_counts:
                                print('Already using all features, skipping...')
                                continue
                            else:
                                print(f"Warning: Requested {n_features} features, but only {len(feature_indices)} available. Using all available features.")
                                n_features = len(feature_indices)
                        
                        # Get the top n features
                        selected_indices = feature_indices[:n_features]
                        
                        # Extract data with selected features
                        X_all_selected = X_all[:, selected_indices]
                        X_test_selected = X_test[:, selected_indices]
                        
                        # Apply PCA if needed
                        if n_components != -1:
                            # Make sure n_components doesn't exceed the number of features or samples
                            n_components_actual = min(n_components, min(n_features, X_all_selected.shape[0]))
                            
                            # Apply PCA
                            pca = PCA(n_components=n_components_actual, random_state=SPLIT_SEED)
                            X_all_pca = pca.fit_transform(X_all_selected)
                            X_test_pca = pca.transform(X_test_selected)
                            
                            # Calculate variance explained
                            explained_variance = np.sum(pca.explained_variance_ratio_) * 100
                        else:
                            pca = None
                            X_all_pca = X_all_selected
                            X_test_pca = X_test_selected
                            explained_variance = 100
                        
                        # Create the model
                        if model_type == 'isolation_forest':
                            percent_pos = np.sum(y_all) / len(y_all)
                            model = ModelConstructor.create_model('numeric', model_type, granularity, n_components, contamination=percent_pos, num_features=n_features)
                        else:
                            model = ModelConstructor.create_model('numeric', model_type, granularity, n_components, num_features=n_features)
                        
                        # Train the model on ALL available data
                        train_start = time.time()
                        model.fit(X_all_pca, y_all)
                        train_end = time.time()
                        print(f'Model {model_type} trained on ALL data in {train_end - train_start:.2f} seconds')
                        
                        # Make predictions on test set (for sanity check metrics only)
                        print('\nGenerating test set predictions for sanity check metrics...')
                        if model_type == 'isolation_forest':
                            y_test_pred = model.predict(X_test_pca)
                            y_test_pred[y_test_pred == 1] = 0
                            y_test_pred[y_test_pred == -1] = 1
                            
                            anomaly_scores = model.decision_function(X_test_pca)
                            y_test_pred_proba = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
                        elif model_type == 'nn_simple' or model_type == 'nn_complex':
                            y_test_pred_proba = model.predict(X_test_pca)
                            y_test_pred = (y_test_pred_proba > 0.5).astype(int)
                        else:
                            y_test_pred = model.predict(X_test_pca)
                            y_test_pred_proba = model.predict_proba(X_test_pca)[:, 1]
                        
                        # Evaluate on test set (sanity check only - model was trained on this data too)
                        print('NOTE: Test metrics are for sanity check only - model was trained on test data too')
                        test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba, "Test (Sanity Check)")
                        
                        # Store this configuration's results
                        result_row = {
                            'granularity': granularity,
                            'model_type': model_type,
                            'oversampling_ratio': oversampling_ratio,
                            'n_features': n_features,
                            'n_components': n_components,
                            'test_accuracy_sanity': test_metrics['accuracy'],
                            'test_precision_sanity': test_metrics['precision'],
                            'test_recall_sanity': test_metrics['recall'],
                            'test_f1_sanity': test_metrics['f1'],
                            'test_auc_sanity': test_metrics['auc'],
                            'test_hss_sanity': test_metrics['hss'],
                            'test_tss_sanity': test_metrics['tss'],
                            'test_cm_sanity': test_metrics['confusion_matrix'],
                            'split_seed': SPLIT_SEED,
                            'selected_features': selected_features[:n_features],
                            'explained_variance': explained_variance,
                            'trained_on_all_data': True
                        }
                        
                        all_results.append(result_row)
                        
                        # Save the final operational model, scaler, pca, and feature indices
                        final_model_data = {
                            'model': model,
                            'scaler': scaler,
                            'pca': pca,
                            'feature_indices': selected_indices,
                            'feature_names': selected_features[:n_features],
                            'test_metrics_sanity_check': test_metrics,
                            'training_data_size': len(X_all),
                            'training_sep_count': np.sum(y_all),
                            'trained_on_all_data': True,
                            'note': 'Model trained on ALL available data including test set'
                        }
                        
                        joblib.dump(final_model_data, f'{RESULTS_DIR}/{model_name}_final_model_data.joblib')
                        print(f"\nFinal operational model saved to {RESULTS_DIR}/{model_name}_final_model_data.joblib")
                        
                        # Save intermediate results (excluding numpy arrays)
                        results_df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} 
                                                 for row in all_results])
                        results_df.to_csv(f'{RESULTS_DIR}/{NAME}_final_results_so_far.csv', index=False)
    
    # Convert results to DataFrame, excluding numpy arrays and lists
    results_df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} 
                             for row in all_results])
    
    # Save results
    results_df.to_csv(f'{RESULTS_DIR}/{NAME}_final_results.csv', index=False)
    
    # Save complete results including confusion matrices
    joblib.dump(all_results, f'{RESULTS_DIR}/{NAME}_final_results_with_cms.joblib')

    if len(all_results) == 0:
        print("No valid results found. Exiting.")
        return
        
    # Get best configuration based on F1 score (should only be one if following current params)
    best_idx = results_df['test_f1_sanity'].idxmax()
    best_config = results_df.loc[best_idx].to_dict()
    best_complete_config = all_results[best_idx]
    
    # Print best configuration
    print("\nFinal Model Configuration:")
    print(f"Model type: {best_config['model_type']}")
    print(f"Granularity: {best_config['granularity']}")
    print(f"Oversampling Ratio: {best_config['oversampling_ratio']}")
    print(f"Number of Features: {best_config['n_features']}")
    print(f"Selected Features: {best_config['selected_features']}")
    print(f"Number of PCA Components: {best_config['n_components']}")
    print(f"Explained Variance: {best_config['explained_variance']:.2f}%")
    print(f"Trained on ALL data: {best_config['trained_on_all_data']}")
    
    def get_f_half(precision, recall):
        return (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
    
    # Print all test metrics (sanity check only)
    print(f"\nTest Set Performance (Sanity Check - model was trained on this data too):")
    print(f"Test Accuracy: {best_config['test_accuracy_sanity']:.4f}")
    print(f"Test Precision: {best_config['test_precision_sanity']:.4f}")
    print(f"Test Recall: {best_config['test_recall_sanity']:.4f}")
    print(f"Test F1 Score: {best_config['test_f1_sanity']:.4f}")
    print(f"Test F-0.5 Score: {get_f_half(best_config['test_precision_sanity'], best_config['test_recall_sanity']):.4f}")
    print(f"Test AUC: {best_config['test_auc_sanity']:.4f}")
    print(f"Test HSS: {best_config['test_hss_sanity']:.4f}")
    print(f"Test TSS: {best_config['test_tss_sanity']:.4f}")
    print(f"Test Confusion Matrix:")
    print(best_complete_config['test_cm_sanity'])
    
    # Get final model name
    final_model_name = f"{NAME}_{best_config['granularity']}_{best_config['oversampling_ratio']}_{best_config['model_type']}_{best_config['n_features']}_{best_config['n_components']}_alldata"
    
    # Save the final configuration metadata
    final_model_metadata = {
        'model_type': best_config['model_type'],
        'granularity': best_config['granularity'],
        'oversampling_ratio': best_config['oversampling_ratio'],
        'n_features': best_config['n_features'],
        'n_components': best_config['n_components'],
        'explained_variance': best_config['explained_variance'],
        'test_metrics_sanity_check': {
            'accuracy': best_config['test_accuracy_sanity'],
            'precision': best_config['test_precision_sanity'],
            'recall': best_config['test_recall_sanity'],
            'f1': best_config['test_f1_sanity'],
            'auc': best_config['test_auc_sanity'],
            'hss': best_config['test_hss_sanity'],
            'tss': best_config['test_tss_sanity'],
            'confusion_matrix': best_complete_config['test_cm_sanity'].tolist()
        },
        'model_path': f'{final_model_name}_final_model_data.joblib',
        'split_seed': SPLIT_SEED,
        'trained_on_all_data': True,
        'training_note': 'Model trained on ALL available data (train+validation+test) for maximum operational performance',
        'metrics_note': 'Test metrics are sanity check only - model was trained on test data too'
    }
    
    # Save metadata
    joblib.dump(final_model_metadata, f'{RESULTS_DIR}/{NAME}_final_model_metadata.joblib')
    
    print(f"\nFinal operational model configuration and metadata saved to {RESULTS_DIR}/ directory")
    print(f"Model file: {final_model_name}_final_model_data.joblib")
    print(f"Metadata file: {NAME}_final_model_metadata.joblib")
    print(f"\nThis model is trained on ALL available data (train+val+test) and ready for operational use.")
    print(f"Test metrics shown above are for sanity check only since model was trained on test data too.")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)')

if __name__ == "__main__":
    main()