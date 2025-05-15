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
import os
import random
from sklearn.utils import shuffle
import sys
import pathlib
import json
from datetime import timedelta
import datetime

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir, 'SEPPrediction'))

from ModelConstructor import ModelConstructor

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Best hyperparameters by data type
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
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.5), 1)  # Reduce CPU usage to avoid overload with many models
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

def prepare_data(train_df, test_df, granularity, data_loader_module, oversampling_ratio):
    """
    Prepare data for training and evaluation
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        granularity: Data granularity
        data_loader_module: Imported data loader module
        oversampling_ratio: Ratio for oversampling positive class
        
    Returns:
        X_train_pca, y_train, X_test_pca, y_test, model_data: Prepared data and model artifacts
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
    test_generator = SEPInputDataGenerator(
        test_df, batch_size, False, granularity, 
        use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2
    )
    
    # Extract all data from generators
    print('\nExtracting data from generators...')
    X_train, y_train, _ = extract_all_data(train_generator)
    X_test, y_test, dts_test = extract_all_data(test_generator)

    print('X_train type:', type(X_train), 'X_train shape:', X_train.shape)
    print('y_train type:', type(y_train), 'y_train shape:', y_train.shape)
    
    # Check for NaN and Inf values
    print('\nChecking for NaN and Inf values:')
    print(f'Train NaN count: {np.isnan(X_train).sum()}, Inf count: {np.isinf(X_train).sum()}')
    print(f'Test NaN count: {np.isnan(X_test).sum()}, Inf count: {np.isinf(X_test).sum()}')
    
    # Replace any NaN or Inf values with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply class balancing to training set
    print('\nBefore resampling:')
    print('Train set count:', len(X_train))
    print('Train set SEP count:', np.sum(y_train))
    print('Test set count:', len(X_test))
    print('Test set SEP count:', np.sum(y_test))
    
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
    X_test_selected = X_test[:, selected_indices]
    
    # Apply PCA if needed
    if component_count != -1:
        # Make sure n_components doesn't exceed the number of features or samples
        n_components_actual = min(component_count, min(n_features, X_train_selected.shape[0]))
        
        # Apply PCA
        pca = PCA(n_components=n_components_actual, random_state=SEED)
        X_train_pca = pca.fit_transform(X_train_selected)
        X_test_pca = pca.transform(X_test_selected)
        
        # Calculate variance explained
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f'PCA explained variance: {explained_variance:.2f}%')
    else:
        pca = None
        X_train_pca = X_train_selected
        X_test_pca = X_test_selected
    
    # Store model data artifacts
    model_data = {
        'scaler': scaler,
        'pca': pca,
        'feature_indices': selected_indices,
        'feature_names': selected_feature_names
    }
    
    return X_train_pca, y_train_resampled, X_test_pca, y_test, model_data, dts_test

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type, data_type, model_data):
    """
    Train and evaluate a model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_type: Type of model to train
        data_type: Type of data (photospheric, coronal, or numeric)
        model_data: Dictionary with model artifacts
        
    Returns:
        model, prediction, probability: Trained model, prediction and probability for test point
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
    print(f'\nTraining {model_type} model...')
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    print(f'Model trained in {train_end - train_start:.2f} seconds')
    
    # Make predictions on test point
    prediction = model.predict(X_test)[0]
    probability = model.predict_proba(X_test)[0][1]  # Probability of positive class
    
    # Store model in model_data
    model_data['model'] = model
    
    return model, prediction, probability

def main(data_type, train_df, test_df, output_dir=None):
    """
    Main function to train and evaluate a model for a single hold-out point
    
    Args:
        data_type: Type of data ('photospheric', 'coronal', or 'numeric')
        train_df: Training DataFrame
        test_df: Test DataFrame (should contain only one row)
        output_dir: Directory to save results (optional)
        
    Returns:
        prediction, probability: The prediction (0/1) and probability for the test point
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Check that test_df has only one row
    if len(test_df) != 1:
        print(f"Warning: Test dataframe has {len(test_df)} rows, expected 1. Using first row.")
        test_df = test_df.iloc[[0]]
    
    # Check data_type is valid
    if data_type not in ['photospheric', 'coronal', 'numeric']:
        raise ValueError(f"Invalid data type: {data_type}. Must be one of 'photospheric', 'coronal', or 'numeric'")
    
    # Get the best parameters for this data type
    best_params = BEST_PARAMS[data_type]
    
    # Dynamically import the appropriate data loader module
    if data_type == 'photospheric':
        import PhotosphericDataLoader as data_loader_module
    elif data_type == 'coronal':
        import CoronalDataLoader as data_loader_module
    elif data_type == 'numeric':
        import NumericDataLoader as data_loader_module
    
    print(f"Training data: {len(train_df)} rows")
    print(f"Test data: {len(test_df)} rows")
    
    # Preprocess the data
    print("Preprocessing data...")
    
    # Make sure 'Produced an SEP' column exists, create if not
    if 'Produced an SEP' not in train_df.columns:
        train_df['Produced an SEP'] = (train_df['Number of SEPs Produced'] > 0) * 1
    if 'Produced an SEP' not in test_df.columns:
        test_df['Produced an SEP'] = (test_df['Number of SEPs Produced'] > 0) * 1
    
    # Make sure 'Year' column exists, create if not
    if 'Year' not in train_df.columns and 'Filename General' in train_df.columns:
        train_df['Year'] = train_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
    if 'Year' not in test_df.columns and 'Filename General' in test_df.columns:
        test_df['Year'] = test_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
    
    # Make sure 'Is Plage' column is an integer
    if 'Is Plage' in train_df.columns:
        train_df['Is Plage'] = train_df['Is Plage'].astype(int)
    if 'Is Plage' in test_df.columns:
        test_df['Is Plage'] = test_df['Is Plage'].astype(int)
    
    # Make sure 'Most Probable AR Num' column exists, create if not
    if 'Most Probable AR Num' not in train_df.columns and 'Relevant Active Regions' in train_df.columns:
        train_df['Most Probable AR Num'] = train_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
    if 'Most Probable AR Num' not in test_df.columns and 'Relevant Active Regions' in test_df.columns:
        test_df['Most Probable AR Num'] = test_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
    
    # Prepare data
    print("Preparing data for model training...")
    X_train, y_train, X_test, y_test, model_data, dts_test = prepare_data(
        train_df, test_df, 
        best_params['GRANULARITY'], 
        data_loader_module, 
        best_params['OVERSAMPLING_RATIO']
    )
    
    # Train and evaluate model
    model, prediction, probability = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        best_params['MODEL_TYPE'], 
        data_type, 
        model_data
    )
    
    # Save the model if output_dir is specified
    if output_dir:
        # Save feature importance data
        if hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'Feature': model_data['feature_names'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            feature_importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        # Generate prediction JSON file
        dt_str = dts_test[0]  # Should only be one test point
        dt = pd.to_datetime(dt_str, format='%Y%m%d_%H%M%S_TAI')
        createSepJson4ccmc(dt, probability, prediction, output_dir)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'Model training and evaluation completed in {total_time:.2f} seconds')
    
    return prediction, probability

def createSepJson4ccmc(input_dt, DiskProb, prediction, output_dir):
    """
    Create SPE JSON file for CCMC
    """
    year = input_dt.year
    month = input_dt.month
    day = input_dt.day
    hour = input_dt.hour
    minute = input_dt.minute

    dt_str = datetime.datetime.strftime(input_dt, '%Y-%m-%dT%H:%M:%S')

    # SPEprobValue = 1.00-float(DiskProb[4])/100.00 # SPE probabilty value
    SPEprobValue = DiskProb # No post-processing of the probability value like in empirical MagPy

    forecastEndTime = input_dt + timedelta(days=1)
    forecastEndTimeStr = datetime.datetime.strftime(forecastEndTime, '%Y-%m-%dT%H:%M:%S')
    issueTime = datetime.datetime.now(datetime.timezone.utc)
    issueTimeStr = datetime.datetime.strftime(issueTime, '%Y-%m-%dT%H:%M:%S')
    
    res={
        "sep_forecast_submission": {
            "model":{
                "short_name": "MagPy_ML_SHARP_HMI_CEA_HoldOut1",
                "spase_id": "spase://CCMC/SimulationModel/MagPy-ML/v1"
            },
            "mode": "forecast", 
            "issue_time":f"{issueTimeStr}Z",
            "inputs":[
                {
                    "magnetogram":{
                        "observatory":"SDO",
                        "instrument":"HMI",
                        "products":[
                            {
                                "product":"hmi.sharp_cea_720s_nrt",
                                "last_data_time":f'{year}-{month}-{day}T{hour}:{minute}Z',
                            }
                        ]
                    }
                }
            ],
            "forecasts":[
                {
                    "energy_channel": {
                        "min": 10,
                        "max": -1,
                        "units": "MeV"
                    },
                    "species": "proton",
                    "location": "earth",
                    "prediction_window":{
                        "start_time":f"{dt_str}Z",
                        "end_time":f"{forecastEndTimeStr}Z",
                    },
                    "probabilities":[
                        {
                            "probability_value":f"{SPEprobValue:.5f}",
                            "threshold":10,
                            "threshold_units":"pfu"
                        }
                    ],
                    "all_clear":{
                        "all_clear_boolean": not prediction,
                        "threshold":10,
                        "threshold_units":"pfu",
                        "probability_threshold":0.5
                    }
                }
            ]
        }
    }
    
    # Create directory for JSON output
    json_dir = os.path.join(output_dir, 'JSON_CCMC', 'SEP_JSON')
    os.makedirs(json_dir, exist_ok=True)
    
    # Save JSON file
    MagPyJSONCCMC = f'MagPy-ML-HMI-SHARP-Vector.{year}{month}{day}T{hour}{minute}.{issueTime.year}{issueTime.month:02d}{issueTime.day:02d}T{issueTime.hour:02d}{issueTime.minute:02d}.json'
    with open(os.path.join(json_dir, MagPyJSONCCMC), 'w', encoding='utf8') as JSONFile:
        json.dump(res, JSONFile, indent=2, separators=(',', ': '))

if __name__ == "__main__":
    main('photospheric', None, None)
