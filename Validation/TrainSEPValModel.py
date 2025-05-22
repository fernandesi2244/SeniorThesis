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
import sys
import pathlib
import json
from datetime import timedelta
import datetime

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir, 'SEPPrediction'))

from ModelConstructor import ModelConstructor

# Create output directory for results
RESULTS_DIR = f'results/sep_model_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    
    # Calculate F0.5 score (more weight on precision)
    f_half = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    # Print metrics
    print(f'{set_name} Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'F0.5 Score: {f_half:.4f}')
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
        'f_half': f_half,
        'auc': auc,
        'hss': hss,
        'tss': tss,
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
    
    # Save feature importance
    feature_importance_df.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)
    
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

    print('X_train first 5 rows:', X_train[:5])
    print('y_train first 5 rows:', y_train[:5])
    
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
        model, test_metrics: Trained model and evaluation metrics
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
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Save the model and artifacts
    model_data['model'] = model
    model_data['test_metrics'] = test_metrics
    
    filename = f'{RESULTS_DIR}/{data_type}_{model_type}_model.joblib'
    joblib.dump(model_data, filename)
    print(f'Model saved to {filename}')
    
    return model, test_metrics

def main(data_type, train_df, test_df, output_dir=None):
    """
    Main function to train and evaluate a model
    
    Args:
        data_type: Type of data ('photospheric', 'coronal', or 'numeric')
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save results (optional)
        
    Returns:
        confusion_matrix: The confusion matrix from model evaluation
    """
    # Update output directory if provided
    global RESULTS_DIR
    if output_dir:
        RESULTS_DIR = output_dir
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting SEP model training and evaluation at {time.ctime()}")
    
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
    
    # File paths for saving/loading data
    data_file = f"{RESULTS_DIR}/{data_type}_prepared_data.npz"
    
    if os.path.exists(data_file):
        print(f"Loading prepared data from {data_file}...")
        data = np.load(data_file, allow_pickle=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        model_data = data['model_data'].item()
        dts_test = data['dts_test']
    else:
        print("Preparing data...")
        X_train, y_train, X_test, y_test, model_data, dts_test = prepare_data(
            train_df, test_df, 
            best_params['GRANULARITY'], 
            data_loader_module, 
            best_params['OVERSAMPLING_RATIO']
        )
        print(f"Saving prepared data to {data_file}...")
        np.savez_compressed(
            data_file, 
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test, 
            model_data=model_data, 
            dts_test=dts_test
        )
    
    # Train and evaluate model
    model, test_metrics = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        best_params['MODEL_TYPE'], 
        data_type, 
        model_data
    )

    # Now go through each of the test data and generate a JSON prediction file for each
    for i in range(len(X_test)):
        dt_str = dts_test[i] # Already in the correct format of '%Y%m%d_%H%M%S_TAI'
        dt = pd.to_datetime(dt_str, format='%Y%m%d_%H%M%S_TAI')

        prediction = model.predict(X_test[i].reshape(1, -1))[0]
        prediction_proba = model.predict_proba(X_test[i].reshape(1, -1))[0][1]

        createSepJson4ccmc(dt, prediction_proba, prediction)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nModel training and evaluation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    
    # Return the confusion matrix
    return test_metrics['confusion_matrix']

def createSepJson4ccmc(input_dt, DiskProb, prediction, issueTime=None):
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
    if issueTime is None:
        issueTime = datetime.datetime.now(datetime.timezone.utc)
    issueTimeStr = datetime.datetime.strftime(issueTime, '%Y-%m-%dT%H:%M:%S')
    
    res={
        "sep_forecast_submission": {
            "model":{
                "short_name": "MagPy_ML_SHARP_HMI_CEA",
                "spase_id": "spase://CCMC/SimulationModel/MagPy-ML/v1"
            },
            "mode": "forecast", # not really a forecast though since not predictions not causal (based on future data too, though most likely unrelated)
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
    
    prediction_json_dir = os.path.join('JSON CCMC','SEP JSON')
    if not os.path.exists(prediction_json_dir):
        os.makedirs(prediction_json_dir)
    
    MagPyJSONCCMC = f'MagPy-ML-HMI-SHARP-Vector.{year:04d}{month:02d}{day:02d}T{hour:02d}{minute:02d}.{issueTime.year:04d}{issueTime.month:02d}{issueTime.day:02d}T{issueTime.hour:02d}{issueTime.minute:02d}.json'
    with open(os.path.join(prediction_json_dir, MagPyJSONCCMC), 'w',encoding='utf8') as JSONFile:
        json.dump(res, JSONFile, indent=2, separators=(',', ': '))

if __name__ == "__main__":
    conf_matrix = main()
    print("Final confusion matrix:")
    print(conf_matrix)