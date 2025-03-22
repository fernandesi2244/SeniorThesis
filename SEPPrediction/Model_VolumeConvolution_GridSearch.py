from VolumeSlicesAndCubeDataLoader import SEPInputDataGenerator
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

NAME = 'sep_prediction_volume_convolution_grid_search'

# Create output directory for results
RESULTS_DIR = f'results/volume_convolution'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Multiprocessing setup
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

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

def load_data():
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
    
    return train_df, val_df, test_df

def main():
    """Main function to run the combined feature selection and PCA analysis"""
    start_time = time.time()

    train_df_OG, val_df_OG, test_df_OG = load_data()

    X_train_OG = train_df_OG.drop(columns=['Produced an SEP', 'Number of SEPs Produced'])
    y_train_OG = train_df_OG['Produced an SEP']

    X_val = val_df_OG.drop(columns=['Produced an SEP', 'Number of SEPs Produced'])
    y_val = val_df_OG['Produced an SEP']

    X_test = test_df_OG.drop(columns=['Produced an SEP', 'Number of SEPs Produced'])
    y_test = test_df_OG['Produced an SEP']

    # granularities = ['per-blob', 'per-disk-4hr', 'per-disk-1d']
    granularities = ['per-disk-1d']

    oversampling_ratios = [0.65] # [0.1, 0.25, 0.5, 0.65, 0.75, 1] # pos:neg ratio.

    model_types = [
        'conv_nn_on_slices_and_cube',
        'conv_nn_on_cube',
        'conv_nn_on_slices',
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

            # Can generate right now since they don't depend on oversampling ratio
            val_generator = SEPInputDataGenerator(val_df_OG, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
            test_generator = SEPInputDataGenerator(test_df_OG, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)

            for oversampling_ratio in oversampling_ratios:
                print('\n' + '-'*50)
                print(f'\nEvaluating oversampling ratio: {oversampling_ratio}')
                print('-'*50)
            
                # Apply class balancing to training set
                print('\nBefore resampling:')
                print('Train set count:', len(train_df_OG))
                print('Train set SEP count:', np.sum(y_train_OG))

                # if oversampling ratio is less than or equal to the current ratio, skip
                if oversampling_ratio <= np.mean(y_train_OG):
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

                # TODO: Create data loader from oversampled data
                train_df = pd.concat([X_train, y_train], axis=1)
                train_generator = SEPInputDataGenerator(train_df, batch_size=32, shuffle=False, granularity=granularity, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)

                # time model loading
                model = ModelConstructor.create_model('slices_and_cube', model_type, granularity=granularity, n_components=-1)

                # Define some callbacks to improve training.
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
                checkpoint_best_every_50 = tf.keras.callbacks.ModelCheckpoint(
                    f"{NAME}_checkpoint_best_every_50.keras",
                    save_best_only=True,
                    save_freq=50, # save every 50 batches
                )
                checkpoint_every_50 = tf.keras.callbacks.ModelCheckpoint(
                    f"{NAME}_checkpoint_every_50.keras",
                    save_best_only=False,
                    save_freq=50, # save every 50 batches
                )
                
                # time model training
                train_start = time.time()
                # model.fit(X_train, y_train)

                model.fit(train_generator, epochs=10, validation_data=val_generator,
                            callbacks=[early_stopping, reduce_lr, checkpoint_best_every_50, checkpoint_every_50])

                train_end = time.time()
                print(f'Model trained in {train_end - train_start:.2f} seconds')
                
                # Make predictions
                y_pred_proba = model.predict(X_val)
                # Assume 0.5 threshold for now, prob need to optimize over too
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Validation")
                
                # Extract metrics to store in results list
                # TODO: Update this and later with all selected hyperparameters
                result_row = {
                    'oversampling_ratio': oversampling_ratio,
                    'model_type': model_type,
                    'model': model,
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

    # Get best configuration
    best_config = results_df.loc[results_df['f1'].idxmax()].to_dict()
    
    # Print best configuration
    print("\nBest Configuration:")
    print(f"Model type: {best_config['model_type']}")
    print(f"Oversampling Ratio: {best_config['oversampling_ratio']}")
    print(f"F1 Score: {best_config['f1']:.4f}")
    print(f"Accuracy: {best_config['accuracy']:.4f}")
    print(f"Precision: {best_config['precision']:.4f}")
    print(f"Recall: {best_config['recall']:.4f}")
    print(f"AUC: {best_config['auc']:.4f}")
    
    # Now, evaluate the best model on the test set
    print("\nEvaluating best model on test set...")
    
    # Make predictions
    y_test_pred_proba = best_config['model'].predict(X_test)
    # Assume 0.5 threshold for now, prob need to optimize over too
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Evaluate performance
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Save the best model, PCA transformer, and feature indices
    model_metadata = {
        'model_type': best_config['model_type'],
        'oversampling_ratio': best_config['oversampling_ratio'],
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
    joblib.dump(model_metadata, f'{RESULTS_DIR}/{NAME}_metadata.joblib')
    
    print(f"\nBest model and metadata saved to {RESULTS_DIR}/ directory")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nAnalysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')

if __name__ == "__main__":
    main()
