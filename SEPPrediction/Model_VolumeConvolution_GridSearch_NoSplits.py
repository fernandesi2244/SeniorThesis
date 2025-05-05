from VolumeSlicesAndCubeDataLoader import PrimarySEPInputDataGenerator
from VolumeSlicesAndCubeDataLoader import SecondarySEPInputDataGenerator as SC_SecondarySEPInputDataGenerator
from VolumeCubeDataLoader import SecondarySEPInputDataGenerator as C_SecondarySEPInputDataGenerator
from VolumeSlicesDataLoader import SecondarySEPInputDataGenerator as S_SecondarySEPInputDataGenerator
from VolumeFullCubeDataLoader import SecondarySEPInputDataGenerator as FC_SecondarySEPInputDataGenerator

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

SEED_TO_USE = 42

NAME = f'sep_prediction_volume_convolution_grid_search_{SEED_TO_USE}'

# Create output directory for results
RESULTS_DIR = f'results/volume_convolution'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

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

    """
    HSS = 2 * (TP * TN - FN * FP)/((TP + FN) * (FN + TN) + (TP + FP) * (TN + FP))
    TSS = TP / (TP + FN) - FP / (FP + TN)
    """
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
    print(f'TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}')
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'hss': hss,
        'tss': tss,
        'confusion_matrix': cm,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP
    }

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

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Otherwise, we need to load the data from the original CSV files and preprocess it
    blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv'
    blob_df = pd.read_csv(blob_df_filename)

    # # Go through all the data and make sure slices and cubes exist for all of them. if they don't, exclude
    # # them from the dataset. This is because we can't train on them if they don't exist.
    # rows_to_drop = []
    # for i, row in blob_df.iterrows():
    #     filename_general = row['Filename General']
    #     blob_index = row['Blob Index']

    #     xy_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xy.npy')
    #     xz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xz.npy')
    #     yz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_yz.npy')
    #     cube_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_cube.npy')

    #     if not os.path.exists(xy_slices_path) or not os.path.exists(xz_slices_path) or not os.path.exists(yz_slices_path) or not os.path.exists(cube_path):
    #         print(f'Warning: Missing slices or cube for {filename_general} blob {blob_index}. Dropping from dataset.')
    #         rows_to_drop.append(i)
    
    # if len(rows_to_drop) > 0:
    #     blob_df.drop(rows_to_drop, inplace=True)

    # blob_df.reset_index(drop=True, inplace=True)

    # print('Dropped', len(rows_to_drop), 'rows due to missing slices or cube.')

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
                grouped_train[grouped_train == 0].index, test_size=0.25, random_state=SEED_TO_USE
            )
            train_regions = np.concatenate([min_train_regions, remaining_train_regions])
        else:
            train_regions, val_regions = train_test_split(
                grouped_train.index, test_size=0.25, stratify=grouped_train, random_state=SEED_TO_USE
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

    # Save the data for future use
    np.save(train_features_file, X_train)
    np.save(train_labels_file, y_train)

    np.save(val_features_file, X_val)
    np.save(val_labels_file, y_val)

    np.save(test_features_file, X_test)
    np.save(test_labels_file, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    """Main function to run the combined feature selection and PCA analysis"""
    start_time = time.time()
    print(f'Starting analysis for SEP prediction using volume convolution at {time.ctime()}')

    # granularities = ['per-blob', 'per-disk-4hr', 'per-disk-1d']
    granularities = ['per-disk-1d', 'per-disk-4hr'] # per-blob not coded in ModelConstructor yet

    oversampling_ratios = [0.75] # pos:neg ratio.

    model_types = [
        # 'conv_nn_on_slices_and_cube',
        # 'conv_nn_on_cube',
        # 'conv_nn_on_cube_simple',
        # 'conv_nn_on_cube_complex',
        # 'conv_nn_on_slices',
        'conv_nn_on_volume'
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

            # Load the model
            if model_type == 'conv_nn_on_slices_and_cube':
                dataloader_type = 'slices_and_cube'
            elif model_type == 'conv_nn_on_cube' or model_type == 'conv_nn_on_cube_simple' or model_type == 'conv_nn_on_cube_complex':
                dataloader_type = 'cube'
            elif model_type == 'conv_nn_on_slices':
                dataloader_type = 'slices'
            elif model_type == 'conv_nn_on_volume':
                dataloader_type = 'volume'
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            model = ModelConstructor.create_model(dataloader_type, model_type, granularity, -1)

            # Load the data
            X_train_OG, y_train_OG, X_val, y_val, X_test, y_test = load_data(granularity)

            # # Check for NaN and Inf values
            # print('\nChecking for NaN and Inf values:')
            # print('X_train_OG:', X_train_OG)
            # print('Shape of X_train_OG:', X_train_OG.shape)
            # print('Type of X_train_OG:', type(X_train_OG))
            # print('Shape of y_train_OG:', y_train_OG.shape)
            # print('Type of y_train_OG:', type(y_train_OG))
            # print(f'Train NaN count: {np.isnan(X_train_OG).sum()}, Inf count: {np.isinf(X_train_OG).sum()}')
            # print(f'Val NaN count: {np.isnan(X_val).sum()}, Inf count: {np.isinf(X_val).sum()}')
            # print(f'Test NaN count: {np.isnan(X_test).sum()}, Inf count: {np.isinf(X_test).sum()}')

            # # Replace any NaN or Inf values with 0. NOTE: This code should never change the data since there are no NaNs.
            # X_train_OG = np.nan_to_num(X_train_OG, nan=0.0, posinf=0.0, neginf=0.0)
            # X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            # X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Standardize the features
            scaler = StandardScaler()
            n_columns = X_train_OG.shape[1]
            columns_to_standardize = list(range(n_columns - 2))  # All columns except the last two (first is either a FilenameGeneral or a list of them depending on the granularity and second is equivalent but for blob index)

            X_train_OG[:, columns_to_standardize] = scaler.fit_transform(X_train_OG[:, columns_to_standardize])
            X_val[:, columns_to_standardize] = scaler.transform(X_val[:, columns_to_standardize])
            X_test[:, columns_to_standardize] = scaler.transform(X_test[:, columns_to_standardize])

            for oversampling_ratio in oversampling_ratios:
                print('\n' + '-'*50)
                print(f'\nEvaluating oversampling ratio: {oversampling_ratio}')
                print('-'*50)
            
                if oversampling_ratio != -1:
                    # Apply class balancing to training set
                    print('\nBefore resampling:')
                    print('Train set count:', len(X_train_OG))
                    print('Train set SEP count:', np.sum(y_train_OG))

                    # if oversampling ratio is less than or equal to the current ratio, skip
                    if oversampling_ratio <= np.sum(y_train_OG) / (len(y_train_OG) - np.sum(y_train_OG)):
                        print(f"Skipping oversampling ratio {oversampling_ratio} as positive class already significant enough.")
                        continue
                    
                    # Over-sample minority class
                    ros = RandomOverSampler(sampling_strategy=oversampling_ratio/2, random_state=SEED_TO_USE)
                    X_train, y_train = ros.fit_resample(X_train_OG, y_train_OG)
                    
                    # Under-sample majority class
                    rus = RandomUnderSampler(sampling_strategy=oversampling_ratio, random_state=SEED_TO_USE)
                    X_train, y_train = rus.fit_resample(X_train, y_train)

                    # Reshuffle the data
                    X_train, y_train = shuffle(X_train, y_train, random_state=SEED_TO_USE)
                    
                    print('After resampling:')
                    print('Train set count:', len(X_train))
                    print('Train set SEP count:', np.sum(y_train))
                    
                    # Print dataset statistics
                    print('\nDataset Statistics:')
                    print(f'Train set size: {len(X_train)}, SEP events: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)')
                    print(f'Validation set size: {len(X_val)}, SEP events: {np.sum(y_val)} ({np.mean(y_val)*100:.2f}%)')
                    print(f'Test set size: {len(X_test)}, SEP events: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)')
                    print(f'Number of features: {X_train.shape[1]}')
                else:
                    print('\nNo resampling but printing stats:')
                    print('Train set count:', len(X_train_OG))
                    print('Train set SEP count:', np.sum(y_train_OG))

                    X_train = X_train_OG
                    y_train = y_train_OG

                    X_train, y_train = shuffle(X_train, y_train, random_state=SEED_TO_USE)

                    # Print one record from the training set
                    print('Train set example:', X_train[0])
                    print('Train set label:', y_train[0])

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
                elif model_type == 'conv_nn_on_volume':
                    train_generator = FC_SecondarySEPInputDataGenerator(train_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                    val_generator = FC_SecondarySEPInputDataGenerator(val_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                    test_generator = FC_SecondarySEPInputDataGenerator(test_arr, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
                else:
                    raise ValueError(f"Invalid model type: {model_type}")

                # Print one record from the first batch of the training set
                X_batch, y_batch = train_generator[0]
                print('Train set example shape:', X_batch[0].shape)
                print('Train set label:', y_batch[0])

                # Define some callbacks to improve training.
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
                checkpoint_best_every_50 = tf.keras.callbacks.ModelCheckpoint(
                    f"{NAME}_{granularity}_{oversampling_ratio}_{model_type}_checkpoint_best_every_50.keras",
                    save_best_only=True,
                    save_freq=50, # save every 50 batches
                )
                checkpoint_every_50 = tf.keras.callbacks.ModelCheckpoint(
                    f"{NAME}_{granularity}_{oversampling_ratio}_{model_type}_checkpoint_every_50.keras",
                    save_best_only=False,
                    save_freq=50, # save every 50 batches
                )
                
                # time model training
                train_start = time.time()
                # model.fit(X_train, y_train)

                model.fit(train_generator, epochs=10, validation_data=val_generator,
                            callbacks=[early_stopping, reduce_lr, checkpoint_best_every_50, checkpoint_every_50])

                train_end = time.time()
                print(f'Model {model_type} trained in {train_end - train_start:.2f} seconds')
                
                # Make predictions
                y_pred_proba = model.predict(val_generator)
                # Assume 0.5 threshold for now, prob need to optimize over too
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics. NOTE: y_val can be used directly since the generator is deterministic and doesn't shuffle.
                metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Validation")
                
                # Extract metrics to store in results list
                # TODO: Update this and later with all selected hyperparameters
                result_row = {
                    'granularity': granularity,
                    'oversampling_ratio': oversampling_ratio,
                    'model_type': model_type,
                    'model': model,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'auc': metrics['auc'],
                    'hss': metrics['hss'],
                    'tss': metrics['tss'],
                    'TN': metrics['TN'],
                    'FP': metrics['FP'],
                    'FN': metrics['FN'],
                    'TP': metrics['TP'],
                    'test_loader': test_generator,
                    'test_labels': y_test,
                }
                
                all_results.append(result_row)

                # Save current state of all_results
                results_df = pd.DataFrame(all_results)
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
    print(f"AUC: {best_config['auc']:.4f}")
    print(f"HSS: {best_config['hss']:.4f}")
    print(f"TSS: {best_config['tss']:.4f}")
    print(f"TN: {best_config['TN']}, FP: {best_config['FP']}, FN: {best_config['FN']}, TP: {best_config['TP']}")
    
    # Now, evaluate the best model on the test set
    print("\nEvaluating best model on test set...")
    
    # Make predictions
    test_generator = best_config['test_loader']
    y_test = best_config['test_labels']
    y_test_pred_proba = best_config['model'].predict(test_generator)
    # Assume 0.5 threshold for now, prob need to optimize over too
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Evaluate performance
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Save the best model, PCA transformer, and feature indices
    model_metadata = {
        'model_type': best_config['model_type'],
        'oversampling_ratio': best_config['oversampling_ratio'],
        'granularity': best_config['granularity'],
        'train_metrics': {
            'accuracy': best_config['accuracy'],
            'precision': best_config['precision'],
            'recall': best_config['recall'],
            'f1': best_config['f1'],
            'auc': best_config['auc'],
            'hss': best_config['hss'],
            'tss': best_config['tss'],
            'TN': best_config['TN'],
            'FP': best_config['FP'],
            'FN': best_config['FN'],
            'TP': best_config['TP'],
        },
        'test_metrics': test_metrics,
    }
    
    # Save model and metadata
    joblib.dump(best_config['model'], f'{RESULTS_DIR}/{NAME}_best_model.joblib')
    joblib.dump(model_metadata, f'{RESULTS_DIR}/{NAME}_metadata.joblib')
    
    print(f"\nBest model and metadata saved to {RESULTS_DIR}/ directory")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nAnalysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')

if __name__ == "__main__":
    main()