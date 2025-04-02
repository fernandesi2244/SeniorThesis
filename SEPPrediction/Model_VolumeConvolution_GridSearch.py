from VolumeSlicesAndCubeDataLoader import PrimarySEPInputDataGenerator
from VolumeSlicesAndCubeDataLoader import SecondarySEPInputDataGenerator as SC_SecondarySEPInputDataGenerator
from VolumeCubeDataLoader import SecondarySEPInputDataGenerator as C_SecondarySEPInputDataGenerator
from VolumeSlicesDataLoader import SecondarySEPInputDataGenerator as S_SecondarySEPInputDataGenerator

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
import tensorflow as tf

GENERATED_VOLUME_SLICES_AND_CUBE_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/VolumeSlicesAndCubes'

NAME = 'sep_prediction_volume_convolution_grid_search'

# Create output directory for results
RESULTS_DIR = f'results/volume_convolution'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Number of different dataset splits to use
NUM_SPLITS = 5
# Define random seeds for each split
SPLIT_SEEDS = [42, 123, 456, 789, 1010]

# Multiprocessing setup
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

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

def evaluate_model(y_true, y_pred, y_pred_proba, set_name="", cm=None):
    """
    Evaluate model performance with various metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities for positive class
        set_name: Name of the dataset being evaluated (e.g., "Validation", "Test")
        cm: Pre-computed confusion matrix (if None, it will be calculated)
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calculate confusion matrix components
    if cm is None:
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
    
    # Print metrics
    print(f'{set_name} Results from Combined Confusion Matrix:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
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

def load_data(granularity, split_seed):
    """
    Load data for a specific granularity and split seed
    
    Args:
        granularity: Data granularity ('per-blob', 'per-disk-4hr', or 'per-disk-1d')
        split_seed: Random seed for train/test split
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Data arrays
    """
    if granularity == 'per-blob':
        train_features_file = f'{NAME}_X_train_data_per_blob_split{split_seed}.npy'
        train_labels_file = f'{NAME}_y_train_data_per_blob_split{split_seed}.npy'
        val_features_file = f'{NAME}_val_data_per_blob_split{split_seed}.npy'
        val_labels_file = f'{NAME}_val_labels_per_blob_split{split_seed}.npy'
        test_features_file = f'{NAME}_test_data_per_blob_split{split_seed}.npy'
        test_labels_file = f'{NAME}_test_labels_per_blob_split{split_seed}.npy'
    elif granularity == 'per-disk-4hr':
        train_features_file = f'{NAME}_X_train_data_per_disk_4hr_split{split_seed}.npy'
        train_labels_file = f'{NAME}_y_train_data_per_disk_4hr_split{split_seed}.npy'
        val_features_file = f'{NAME}_val_data_per_disk_4hr_split{split_seed}.npy'
        val_labels_file = f'{NAME}_val_labels_per_disk_4hr_split{split_seed}.npy'
        test_features_file = f'{NAME}_test_data_per_disk_4hr_split{split_seed}.npy'
        test_labels_file = f'{NAME}_test_labels_per_disk_4hr_split{split_seed}.npy'
    elif granularity == 'per-disk-1d':
        train_features_file = f'{NAME}_X_train_data_per_disk_1d_split{split_seed}.npy'
        train_labels_file = f'{NAME}_y_train_data_per_disk_1d_split{split_seed}.npy'
        val_features_file = f'{NAME}_val_data_per_disk_1d_split{split_seed}.npy'
        val_labels_file = f'{NAME}_val_labels_per_disk_1d_split{split_seed}.npy'
        test_features_file = f'{NAME}_test_data_per_disk_1d_split{split_seed}.npy'
        test_labels_file = f'{NAME}_test_labels_per_disk_1d_split{split_seed}.npy'
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

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Otherwise, we need to load the data from the original CSV files and preprocess it
    blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv'
    blob_df = pd.read_csv(blob_df_filename)

    # Go through all the data and make sure slices and cubes exist for all of them. if they don't, exclude
    # them from the dataset. This is because we can't train on them if they don't exist.
    rows_to_drop = []
    for i, row in blob_df.iterrows():
        filename_general = row['Filename General']
        blob_index = row['Blob Index']

        xy_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xy.npy')
        xz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xz.npy')
        yz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_yz.npy')
        cube_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_cube.npy')

        if not os.path.exists(xy_slices_path) or not os.path.exists(xz_slices_path) or not os.path.exists(yz_slices_path) or not os.path.exists(cube_path):
            print(f'Warning: Missing slices or cube for {filename_general} blob {blob_index}. Dropping from dataset.')
            rows_to_drop.append(i)
    
    if len(rows_to_drop) > 0:
        blob_df.drop(rows_to_drop, inplace=True)

    blob_df.reset_index(drop=True, inplace=True)

    print('Dropped', len(rows_to_drop), 'rows due to missing slices or cube.')

    print(f"Loaded dataset with {len(blob_df)} rows")

    # Preprocess the data
    blob_df['Produced an SEP'] = (blob_df['Number of SEPs Produced'] > 0) * 1  # 1 if produced, 0 otherwise
    blob_df['Year'] = blob_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
    blob_df['Is Plage'] = blob_df['Is Plage'].astype(int)
    blob_df['Most Probable AR Num'] = blob_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
    
    # Split the data by year to maintain the same structure as in the original scripts
    # But use the provided split_seed for reproducibility
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
                grouped[grouped == 0].index, test_size=0.2, random_state=split_seed
            )
            train_regions = np.concatenate([min_train_regions, remaining_train_regions])
        else:
            train_regions, test_regions = train_test_split(
                grouped.index, test_size=0.2, stratify=grouped, random_state=split_seed
            )
    
        # Select records based on their active region
        train_from_year = blobs_in_year[blobs_in_year['Most Probable AR Num'].isin(train_regions)]
        test_from_year = blobs_in_year[blobs_in_year['Most Probable AR Num'].isin(test_regions)]
    
        # Further split train into train and validation
        grouped_train = train_from_year.groupby('Most Probable AR Num')['Produced an SEP'].max()
        
        if grouped_train[grouped_train == 1].count() == 1:
            min_train_regions = grouped_train[grouped_train == 1].index
            remaining_train_regions, val_regions = train_test_split(
                grouped_train[grouped_train == 0].index, test_size=0.25, random_state=split_seed
            )
            train_regions = np.concatenate([min_train_regions, remaining_train_regions])
        else:
            train_regions, val_regions = train_test_split(
                grouped_train.index, test_size=0.25, stratify=grouped_train, random_state=split_seed
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
    
    # Create data generators to get data at right granularity
    batch_size = 32
    train_generator = PrimarySEPInputDataGenerator(train_df, batch_size=batch_size, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
    val_generator = PrimarySEPInputDataGenerator(val_df, batch_size=batch_size, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
    test_generator = PrimarySEPInputDataGenerator(test_df, batch_size=batch_size, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)

    # Extract all data from the generators
    X_train, y_train = extract_all_data(train_generator)
    X_val, y_val = extract_all_data(val_generator)
    X_test, y_test = extract_all_data(test_generator)

    # Save the data for future use with split_seed in filename
    np.save(train_features_file, X_train)
    np.save(train_labels_file, y_train)

    np.save(val_features_file, X_val)
    np.save(val_labels_file, y_val)

    np.save(test_features_file, X_test)
    np.save(test_labels_file, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    """Main function to run the combined feature selection and PCA analysis with multiple splits"""
    start_time = time.time()
    print(f'Starting analysis for SEP prediction using volume convolution at {time.ctime()}')

    # granularities = ['per-blob', 'per-disk-4hr', 'per-disk-1d']
    granularities = ['per-disk-4hr', 'per-disk-1d'] # per-blob not coded in ModelConstructor yet

    oversampling_ratios = [-1, 0.1, 0.25, 0.5, 0.65, 0.75, 1] # pos:neg ratio.
    # reverse of oversampling_ratios so I can look at better scores first
    oversampling_ratios = [1, 0.75, 0.65, 0.5, 0.25, 0.1, -1] # pos:neg ratio.

    model_types = [
        # 'conv_nn_on_slices_and_cube',
        'conv_nn_on_cube',
        'conv_nn_on_cube_simple',
        'conv_nn_on_cube_complex',
        # 'conv_nn_on_slices',
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

            for oversampling_ratio in oversampling_ratios:
                print('\n' + '-'*50)
                print(f'\nEvaluating oversampling ratio: {oversampling_ratio}')
                print('-'*50)
                
                # Lists to store confusion matrices across splits
                val_cms = []
                test_cms = []
                
                # Train and evaluate models for each split
                for split_idx, split_seed in enumerate(SPLIT_SEEDS):
                    print('\n' + '-'*50)
                    print(f'\nTraining on split {split_idx+1}/{len(SPLIT_SEEDS)} with seed {split_seed}')
                    print('-'*50)
                    
                    # Define model name with split information
                    model_name = f"{NAME}_{granularity}_{oversampling_ratio}_{model_type}_split{split_seed}"
                    
                    # Load the model. Load it here because we want a fresh model for each split.
                    if model_type == 'conv_nn_on_slices_and_cube':
                        dataloader_type = 'slices_and_cube'
                    elif model_type == 'conv_nn_on_cube' or model_type == 'conv_nn_on_cube_simple' or model_type == 'conv_nn_on_cube_complex':
                        dataloader_type = 'cube'
                    elif model_type == 'conv_nn_on_slices':
                        dataloader_type = 'slices'
                    else:
                        raise ValueError(f"Invalid model type: {model_type}")
                    
                    model = ModelConstructor.create_model(dataloader_type, model_type, granularity, -1)

                    # Load the data for this split
                    X_train_OG, y_train_OG, X_val, y_val, X_test, y_test = load_data(granularity, split_seed)

                    # Standardize the features
                    scaler = StandardScaler()
                    n_columns = X_train_OG.shape[1]
                    columns_to_standardize = list(range(n_columns - 2))  # All columns except the last two

                    X_train_OG[:, columns_to_standardize] = scaler.fit_transform(X_train_OG[:, columns_to_standardize])
                    X_val[:, columns_to_standardize] = scaler.transform(X_val[:, columns_to_standardize])
                    X_test[:, columns_to_standardize] = scaler.transform(X_test[:, columns_to_standardize])

                    # Apply resampling if needed
                    if oversampling_ratio != -1:
                        # Apply class balancing to training set
                        print('\nBefore resampling:')
                        print('Train set count:', len(X_train_OG))
                        print('Train set SEP count:', np.sum(y_train_OG))

                        # if oversampling ratio is less than or equal to the current ratio, skip
                        if oversampling_ratio <= np.mean(y_train_OG):
                            print(f"Skipping oversampling ratio {oversampling_ratio} as positive class already significant enough.")
                            continue
                        
                        # Over-sample minority class
                        ros = RandomOverSampler(sampling_strategy=oversampling_ratio/2, random_state=split_seed)
                        X_train, y_train = ros.fit_resample(X_train_OG, y_train_OG)
                        
                        # Under-sample majority class
                        rus = RandomUnderSampler(sampling_strategy=oversampling_ratio, random_state=split_seed)
                        X_train, y_train = rus.fit_resample(X_train, y_train)

                        # Reshuffle the data
                        X_train, y_train = shuffle(X_train, y_train, random_state=split_seed)
                        
                        print('After resampling:')
                        print('Train set count:', len(X_train))
                        print('Train set SEP count:', np.sum(y_train))
                    else:
                        X_train = X_train_OG
                        y_train = y_train_OG
                        X_train, y_train = shuffle(X_train, y_train, random_state=split_seed)

                    # Print dataset statistics
                    print('\nDataset Statistics:')
                    print(f'Train set size: {len(X_train)}, SEP events: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)')
                    print(f'Validation set size: {len(X_val)}, SEP events: {np.sum(y_val)} ({np.mean(y_val)*100:.2f}%)')
                    print(f'Test set size: {len(X_test)}, SEP events: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)')

                    # Create data generators for this split
                    train_arr = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
                    val_arr = np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1)
                    test_arr = np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1)

                    if model_type == 'conv_nn_on_slices_and_cube':
                        train_generator = SC_SecondarySEPInputDataGenerator(train_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                        val_generator = SC_SecondarySEPInputDataGenerator(val_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                        test_generator = SC_SecondarySEPInputDataGenerator(test_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                    elif model_type == 'conv_nn_on_cube' or model_type == 'conv_nn_on_cube_simple' or model_type == 'conv_nn_on_cube_complex':
                        train_generator = C_SecondarySEPInputDataGenerator(train_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                        val_generator = C_SecondarySEPInputDataGenerator(val_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                        test_generator = C_SecondarySEPInputDataGenerator(test_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                    elif model_type == 'conv_nn_on_slices':
                        train_generator = S_SecondarySEPInputDataGenerator(train_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                        val_generator = S_SecondarySEPInputDataGenerator(val_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                        test_generator = S_SecondarySEPInputDataGenerator(test_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                    else:
                        raise ValueError(f"Invalid model type: {model_type}")

                    # Define callbacks for this split
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
                    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
                        f"{model_name}_best.keras",
                        save_best_only=True,
                    )
                    
                    # Train the model for this split
                    train_start = time.time()
                    
                    model.fit(train_generator, epochs=10, validation_data=val_generator,
                              callbacks=[early_stopping, reduce_lr, checkpoint_best])

                    train_end = time.time()
                    print(f'Model trained in {train_end - train_start:.2f} seconds for split {split_idx+1}')
                    
                    # Load the best model for evaluation
                    model = tf.keras.models.load_model(f"{model_name}_best.keras")
                    
                    # Evaluate on validation set
                    val_pred_proba = model.predict(val_generator)
                    val_pred = (val_pred_proba > 0.5).astype(int)
                    val_cm = confusion_matrix(y_val, val_pred)
                    val_cms.append(val_cm)
                    
                    # Save this split's confusion matrix
                    print(f"Validation confusion matrix for split {split_idx+1}:")
                    print(val_cm)
                    
                    # Evaluate on test set
                    test_pred_proba = model.predict(test_generator)
                    test_pred = (test_pred_proba > 0.5).astype(int)
                    test_cm = confusion_matrix(y_test, test_pred)
                    test_cms.append(test_cm)
                    
                    print(f"Test confusion matrix for split {split_idx+1}:")
                    print(test_cm)
                    
                    # Save the model for this split
                    model.save(f"{model_name}.keras")
                
                # End of loop for splits
                
                # Skip this configuration if we didn't get results for all splits
                if len(val_cms) < len(SPLIT_SEEDS):
                    print(f"Skipping configuration due to incomplete splits: {model_type}, {granularity}, {oversampling_ratio}")
                    # This is actually due to skipping oversampling ratio that is too low relative to current balance, which
                    # is possible for a given split. But likely if one split fails, all will fail. So we'll just skip the entire
                    # config with this oversampling ratio.
                    print('!\n'*100)
                    continue
                
                # Combine confusion matrices across all splits
                combined_val_cm = np.sum(np.array(val_cms), axis=0)
                combined_test_cm = np.sum(np.array(test_cms), axis=0)
                
                print("\nCombined validation confusion matrix across all splits:")
                print(combined_val_cm)
                
                print("\nCombined test confusion matrix across all splits:")
                print(combined_test_cm)
                
                # Calculate metrics using the combined confusion matrices
                val_metrics = evaluate_from_combined_confusion_matrix(combined_val_cm, "Validation (Combined)")
                
                # Store this configuration's results
                result_row = {
                    'granularity': granularity,
                    'oversampling_ratio': oversampling_ratio,
                    'model_type': model_type,
                    'accuracy': val_metrics['accuracy'],
                    'precision': val_metrics['precision'],
                    'recall': val_metrics['recall'],
                    'f1': val_metrics['f1'],
                    'hss': val_metrics['hss'],
                    'tss': val_metrics['tss'],
                    'combined_val_cm': combined_val_cm,
                    'combined_test_cm': combined_test_cm,
                }
                
                all_results.append(result_row)
                
                # Save intermediate results
                results_df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} 
                                         for row in all_results])
                results_df.to_csv(f'{RESULTS_DIR}/{NAME}_all_results_so_far.csv', index=False)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(f'{RESULTS_DIR}/{NAME}_all_results.csv', index=False)

    # Get best configuration
    best_config = results_df.loc[results_df['f1'].idxmax()].to_dict()
    
    # Print best configuration
    print("\nBest Configuration:")
    print(f"Model type: {best_config['model_type']}")
    print(f"Oversampling Ratio: {best_config['oversampling_ratio']}")
    print(f"Granularity: {best_config['granularity']}")
    print(f"F1 Score: {best_config['f1']:.4f}")
    print(f"Accuracy: {best_config['accuracy']:.4f}")
    print(f"Precision: {best_config['precision']:.4f}")
    print(f"Recall: {best_config['recall']:.4f}")
    print(f"HSS: {best_config['hss']:.4f}")
    print(f"TSS: {best_config['tss']:.4f}")

    print(f"Combined Validation Confusion Matrix:")
    print(best_config['combined_val_cm'])
    print(f"Combined Test Confusion Matrix:")
    print(best_config['combined_test_cm'])
    
    # Now, evaluate the best model on the test set
    print("\nEvaluating best model on test set...")

    # Print test metrics
    test_cm = best_config['combined_test_cm']
    test_metrics = evaluate_from_combined_confusion_matrix(test_cm, "Test (Combined)")

    print("\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"HSS: {test_metrics['hss']:.4f}")
    print(f"TSS: {test_metrics['tss']:.4f}")
    
    print(f"\nBest model and metadata saved to {RESULTS_DIR}/ directory")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nAnalysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')

if __name__ == "__main__":
    main()
