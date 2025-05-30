"""
Train final SEP models on all available data without holdouts.
This script uses all data to train the optimal models for photospheric, coronal, and numeric data subsets.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import random
from sklearn.utils import shuffle
import pathlib

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir, 'SEPPrediction'))

from ModelConstructor import ModelConstructor

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Best hyperparameters by data type (from validation results)
BEST_PARAMS = {
    'photospheric': {
        'MODEL_TYPE': 'random_forest_complex',
        'GRANULARITY': 'per-disk-4hr',
        'OVERSAMPLING_RATIO': 0.55,
        'FEATURE_COUNT': 90,
        'COMPONENT_COUNT': -1  # No PCA
    },
    'coronal': {
        'MODEL_TYPE': 'random_forest_complex',
        'GRANULARITY': 'per-disk-4hr',
        'OVERSAMPLING_RATIO': 0.7,
        'FEATURE_COUNT': 70,
        'COMPONENT_COUNT': -1  # No PCA
    },
    'numeric': {
        'MODEL_TYPE': 'random_forest_complex',
        'GRANULARITY': 'per-disk-4hr',
        'OVERSAMPLING_RATIO': 0.5,
        'FEATURE_COUNT': 60,
        'COMPONENT_COUNT': -1  # No PCA
    }
}

# Multiprocessing setup
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

def build_feature_names(granularity, data_loader_module):
    """
    Build a list of feature names based on the SEPInputDataGenerator's column definitions.
    
    Args:
        granularity: Data granularity ('per-blob', 'per-disk-4hr', or 'per-disk-1d')
        data_loader_module: The imported data loader module
        
    Returns:
        List of feature names
    """
    SEPInputDataGenerator = data_loader_module.SEPInputDataGenerator
    
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
        dts: Datetimes array
    """
    all_X = []
    all_y = []
    dts = []
    for i in range(len(generator)):
        if i % 100 == 0:
            print(f'Extracting batch {i+1} of {len(generator)}...')
        X_batch, y_batch = generator[i]
        # Extract the datetime from the last element of X_batch
        dt_batch = X_batch[:, -1]  # Assuming the last column contains the timestamps
        # convert to float
        X_batch = X_batch[:, :-1].astype(np.float32) # Remove the last column from X_batch
        all_X.append(X_batch)
        all_y.append(y_batch)
        dts.append(dt_batch)
    
    if len(all_X) > 0:
        return np.vstack(all_X), np.concatenate(all_y), np.concatenate(dts)
    else:
        return np.array([]), np.array([]), np.array([])

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
        random_state=SEED,
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
    
    # Return indices of features sorted by importance
    return feature_indices

def prepare_data(train_df, granularity, data_loader_module, oversampling_ratio):
    """
    Prepare data for training
    
    Args:
        train_df: Training dataframe
        granularity: Data granularity
        data_loader_module: Imported data loader module
        oversampling_ratio: Ratio for oversampling positive class
        
    Returns:
        X_train_pca, y_train, model_data: Prepared data and model artifacts
    """
    # Import SEPInputDataGenerator class from the appropriate module
    SEPInputDataGenerator = data_loader_module.SEPInputDataGenerator
    
    # Create the data generators
    print('\nCreating data generators...')
    batch_size = 64
    train_generator = SEPInputDataGenerator(
        train_df, batch_size, False, granularity, 
        use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2
    )
    
    # Extract all data from generators
    print('\nExtracting data from generators...')
    X_train, y_train, _ = extract_all_data(train_generator)

    print('X_train type:', type(X_train), 'X_train shape:', X_train.shape)
    print('y_train type:', type(y_train), 'y_train shape:', y_train.shape)
    
    # Check for NaN and Inf values
    print('\nChecking for NaN and Inf values:')
    print(f'Train NaN count: {np.isnan(X_train).sum()}, Inf count: {np.isinf(X_train).sum()}')
    
    # Replace any NaN or Inf values with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Apply class balancing to training set
    print('\nBefore resampling:')
    print('Train set count:', len(X_train))
    print('Train set SEP count:', np.sum(y_train))
    
    # Get data type from module name
    data_type = data_loader_module.__name__.replace('DataLoader', '').lower()
    feature_count = BEST_PARAMS[data_type]['FEATURE_COUNT']
    component_count = BEST_PARAMS[data_type]['COMPONENT_COUNT']
    
    # Over-sample minority class
    ros = RandomOverSampler(sampling_strategy=oversampling_ratio/2, random_state=SEED)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    # Under-sample majority class
    rus = RandomUnderSampler(sampling_strategy=oversampling_ratio, random_state=SEED)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_resampled, y_train_resampled)

    # Reshuffle the data
    X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=SEED)
    
    print('After resampling:')
    print('Train set count:', len(X_train_resampled))
    print('Train set SEP count:', np.sum(y_train_resampled))
    
    # Build feature names for interpretation
    feature_names = build_feature_names(granularity, data_loader_module)
    
    # Run feature selection
    feature_indices = feature_selection(X_train_resampled, y_train_resampled, feature_names)
    selected_features = [feature_names[i] for i in feature_indices]
    
    # Get the top n features
    n_features = feature_count
    if n_features > len(feature_indices):
        print(f"Warning: Requested {n_features} features, but only {len(feature_indices)} available. Using all available features.")
        n_features = len(feature_indices)
    
    selected_indices = feature_indices[:n_features]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    # Extract data with selected features
    X_train_selected = X_train_resampled[:, selected_indices]
    
    # Apply PCA if needed
    if component_count != -1:
        # Make sure n_components doesn't exceed the number of features or samples
        n_components_actual = min(component_count, min(n_features, X_train_selected.shape[0]))
        
        # Apply PCA
        pca = PCA(n_components=n_components_actual, random_state=SEED)
        X_train_pca = pca.fit_transform(X_train_selected)
        
        # Calculate variance explained
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f'PCA explained variance: {explained_variance:.2f}%')
    else:
        pca = None
        X_train_pca = X_train_selected
    
    # Store model data artifacts
    model_data = {
        'scaler': scaler,
        'pca': pca,
        'feature_indices': selected_indices,
        'feature_names': selected_feature_names
    }
    
    return X_train_pca, y_train_resampled, model_data

def train_final_model(X_train, y_train, model_type, data_type, model_data):
    """
    Train the final model on all available data
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        data_type: Type of data (photospheric, coronal, or numeric)
        model_data: Dictionary with model artifacts
        
    Returns:
        model: Trained model
    """    
    # Get the best parameters for this data type
    best_params = BEST_PARAMS[data_type]
    
    # Create the model
    model = ModelConstructor.create_model(
        data_type, 
        model_type, 
        best_params['GRANULARITY'], 
        best_params['COMPONENT_COUNT'], 
        num_features=best_params['FEATURE_COUNT']
    )
    
    # Train the model
    print(f'\nTraining final {model_type} model on all data...')
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    print(f'Final model trained in {train_end - train_start:.2f} seconds')
    
    # Save the model and artifacts
    model_data['model'] = model
    
    # Create models directory if it doesn't exist
    os.makedirs('final_models', exist_ok=True)
    
    filename = f'final_models/{data_type}_final_model.joblib'
    joblib.dump(model_data, filename)
    print(f'Final model saved to {filename}')
    
    return model

def train_data_subset(data_type):
    """
    Train a final model for a specific data subset
    
    Args:
        data_type: Type of data ('photospheric', 'coronal', or 'numeric')
    """
    print(f"\n{'='*60}")
    print(f"Training final model for {data_type} data subset")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Get the best parameters for this data type
    best_params = BEST_PARAMS[data_type]
    
    # Dynamically import the appropriate data loader module
    if data_type == 'photospheric':
        import PhotosphericDataLoader as data_loader_module
    elif data_type == 'coronal':
        import CoronalDataLoader as data_loader_module
    elif data_type == 'numeric':
        import NumericDataLoader as data_loader_module
    
    # Load all available data (no holdouts)
    unified_data = pd.read_csv('../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv')
    
    print(f"Total data available: {len(unified_data)} rows")
    
    # Preprocess the data
    print("Preprocessing data...")
    
    # Make sure 'Produced an SEP' column exists, create if not
    if 'Produced an SEP' not in unified_data.columns:
        unified_data['Produced an SEP'] = (unified_data['Number of SEPs Produced'] > 0) * 1
    
    # Make sure 'Year' column exists, create if not
    if 'Year' not in unified_data.columns and 'Filename General' in unified_data.columns:
        unified_data['Year'] = unified_data['Filename General'].apply(lambda x: x.split('.')[3][0:4])
    
    # Make sure 'Is Plage' column is an integer
    if 'Is Plage' in unified_data.columns:
        unified_data['Is Plage'] = unified_data['Is Plage'].astype(int)
    
    # Make sure 'Most Probable AR Num' column exists, create if not
    if 'Most Probable AR Num' not in unified_data.columns and 'Relevant Active Regions' in unified_data.columns:
        unified_data['Most Probable AR Num'] = unified_data['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
    
    print("Preparing data...")
    X_train, y_train, model_data = prepare_data(
        unified_data, 
        best_params['GRANULARITY'], 
        data_loader_module, 
        best_params['OVERSAMPLING_RATIO']
    )
    
    # Train final model
    model = train_final_model(
        X_train, y_train, 
        best_params['MODEL_TYPE'], 
        data_type, 
        model_data
    )
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nFinal model training for {data_type} completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    
    return model

def main():
    """
    Main function to train final models for all data subsets
    """
    print("Training final SEP models on all available data")
    print("=" * 60)
    
    data_subsets = ['photospheric', 'coronal', 'numeric']
    
    for data_subset in data_subsets:
        try:
            train_data_subset(data_subset)
            print(f"✓ Successfully trained final model for {data_subset}")
        except Exception as e:
            print(f"✗ Error training final model for {data_subset}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nAll final models training completed!")
    print("Models saved in the 'final_models' directory.")

if __name__ == "__main__":
    main()
