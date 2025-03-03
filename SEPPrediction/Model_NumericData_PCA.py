from NumericDataLoader import SEPInputDataGenerator
import pandas as pd
import numpy as np
import time
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

NAME = 'sep_prediction_numeric_data_pca'

SELECTED_FEATURE_INDICES = [114, 1, 5, 87, 60, 33, 141, 0, 56, 86, 8, 7, 58, 168, 57, 27, 53, 3, 59, 11]

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

def filter_features_by_indices(X, indices):
    """
    Filter features by indices
    
    Args:
        X: Feature matrix
        indices: List of indices to keep
        
    Returns:
        Filtered feature matrix
    """
    return X[:, indices]

def evaluate_pca_components(X_train, y_train, X_val, y_val, components_list):
    """
    Evaluate different numbers of PCA components using a Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        components_list: List of PCA component counts to test
        
    Returns:
        results_df: DataFrame with evaluation metrics for each component count
    """
    results = []
    
    for n_components in components_list:
        print(f"\nEvaluating PCA with {n_components} components...")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        
        # Calculate variance explained
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        
        # Train a Random Forest classifier
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train_pca, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_val_pca)
        y_pred_proba = rf.predict_proba(X_val_pca)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Store results
        results.append({
            'n_components': n_components,
            'explained_variance': explained_variance,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1': f1
        })
        
        print(f"Variance explained: {explained_variance:.2f}%")
        print(f"Validation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    
    return pd.DataFrame(results)

def plot_pca_evaluation(results_df):
    """
    Plot the evaluation metrics for different PCA component counts
    
    Args:
        results_df: DataFrame with evaluation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot metrics
    ax1.plot(results_df['n_components'], results_df['accuracy'], marker='o', label='Accuracy')
    ax1.plot(results_df['n_components'], results_df['precision'], marker='s', label='Precision')
    ax1.plot(results_df['n_components'], results_df['recall'], marker='^', label='Recall')
    ax1.plot(results_df['n_components'], results_df['auc'], marker='d', label='AUC')
    ax1.plot(results_df['n_components'], results_df['f1'], marker='x', label='F1 Score')
    ax1.set_xlabel('Number of PCA Components')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance vs PCA Components')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot explained variance
    ax2.plot(results_df['n_components'], results_df['explained_variance'], marker='o', color='green')
    ax2.set_xlabel('Number of PCA Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Explained Variance vs PCA Components')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{NAME}_component_evaluation.png', dpi=300)
    plt.close()

def plot_correlation_matrix(X_pca, n_components):
    """
    Plot the correlation matrix of the PCA components
    
    Args:
        X_pca: PCA-transformed data
        n_components: Number of components to display
    """
    # Create a DataFrame with the PCA components
    pca_df = pd.DataFrame(
        X_pca[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Calculate correlation matrix
    corr_matrix = pca_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix of Top {n_components} PCA Components')
    plt.tight_layout()
    plt.savefig(f'{NAME}_pca_correlation_matrix.png', dpi=300)
    plt.close()

def main():
    # Start timer
    start_time = time.time()
    
    # Load dataset
    blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_all_events_including_new_flares_and_TEBBS_fix.csv'
    blob_df = pd.read_csv(blob_df_filename)
    
    print(f"Loaded dataset with {len(blob_df)} rows")
    print(f"Using {len(SELECTED_FEATURE_INDICES)} selected feature indices for PCA")
    
    # Preprocess the data
    blob_df['Produced an SEP'] = (blob_df['Number of SEPs Produced'] > 0) * 1
    blob_df['Year'] = blob_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
    blob_df['Is Plage'] = blob_df['Is Plage'].astype(int)
    blob_df['Most Probable AR Num'] = blob_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
    
    # Split the data by year
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
    
    # Standardize basic columns in the dataframes before passing to the generator
    # This only standardizes the columns that exist in the dataframe, not the derived time series features
    cols_to_scale = SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL + SEPInputDataGenerator.BLOB_ONE_TIME_INFO
    scaler = StandardScaler()
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    # Randomly oversample the majority class and undersample the minority class in the training set for a better balance.
    # Use RandomOversampler w/a sampling strategy of 0.325 and then use RandomUndersampler w/a sampling strategy of 0.65.
    # These "more optimal" ratios were determined from the other NN models from junior year research.
    print('Before resampling:')
    print('Train set count:', len(train_df))
    print('Train set SEP count:', train_df['Produced an SEP'].sum())

    # NOTE: ~37.5x increase in number of SEPs in train set - is quite a huge increase, consider reducing the ratio
    ros = RandomOverSampler(sampling_strategy=0.325)
    train_df, _ = ros.fit_resample(train_df, train_df['Produced an SEP'])

    rus = RandomUnderSampler(sampling_strategy=0.65)
    train_df, _ = rus.fit_resample(train_df, train_df['Produced an SEP'])

    print('After resampling:')
    
    # Print dataset statistics
    print('\nDataset Statistics:')
    print(f'Train set size: {len(train_df)}, SEP events: {train_df["Produced an SEP"].sum()} ({train_df["Produced an SEP"].mean()*100:.2f}%)')
    print(f'Validation set size: {len(val_df)}, SEP events: {val_df["Produced an SEP"].sum()} ({val_df["Produced an SEP"].mean()*100:.2f}%)')
    print(f'Test set size: {len(test_df)}, SEP events: {test_df["Produced an SEP"].sum()} ({test_df["Produced an SEP"].mean()*100:.2f}%)')
    
    # Create data generators
    batch_size = 32
    cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
    
    # Create the data generators
    print('\nCreating data generators...')
    train_generator = SEPInputDataGenerator(train_df, batch_size, False)
    val_generator = SEPInputDataGenerator(val_df, batch_size, False)
    test_generator = SEPInputDataGenerator(test_df, batch_size, False)
    
    print(f'Number of batches in train: {len(train_generator)}')
    print(f'Number of batches in val: {len(val_generator)}')
    print(f'Number of batches in test: {len(test_generator)}')
    
    # Extract all data from generators
    print('\nExtracting data from generators...')
    X_train_full, y_train = extract_all_data(train_generator)
    X_val_full, y_val = extract_all_data(val_generator)
    X_test_full, y_test = extract_all_data(test_generator)
    
    # Extract only the selected feature indices
    X_train = X_train_full[:, SELECTED_FEATURE_INDICES]
    X_val = X_val_full[:, SELECTED_FEATURE_INDICES]
    X_test = X_test_full[:, SELECTED_FEATURE_INDICES]
    
    print(f'Data extraction complete.')
    print(f'Original feature count: {X_train_full.shape[1]}')
    print(f'Selected feature count: {X_train.shape[1]}')
    print(f'Train data shape: {X_train.shape}, labels shape: {y_train.shape}')
    print(f'Validation data shape: {X_val.shape}, labels shape: {y_val.shape}')
    print(f'Test data shape: {X_test.shape}, labels shape: {y_test.shape}')
    
    # Check for NaN and Inf values
    print('\nChecking for NaN and Inf values:')
    print(f'Train NaN count: {np.isnan(X_train).sum()}, Inf count: {np.isinf(X_train).sum()}')
    print(f'Val NaN count: {np.isnan(X_val).sum()}, Inf count: {np.isinf(X_val).sum()}')
    print(f'Test NaN count: {np.isnan(X_test).sum()}, Inf count: {np.isinf(X_test).sum()}')
    
    # Replace any NaN or Inf values with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Test different numbers of PCA components
    # We'll try a range of components based on the input data dimension
    max_components = min(X_train.shape[0], X_train.shape[1])
    component_counts = list(range(5, min(max_components, 100), 5))  # From 5 to min(max_components, 100) in steps of 5
    
    # Add some smaller increments for finer granularity
    component_counts = sorted(list(set(component_counts + [2, 3, 10, 15, 20, 25, 30, 40, 50, 75])))
    component_counts = [c for c in component_counts if c <= max_components]
    
    # Evaluate different PCA components
    print(f"\nEvaluating {len(component_counts)} different PCA component counts...")
    results_df = evaluate_pca_components(X_train, y_train, X_val, y_val, component_counts)
    
    # Save results
    results_df.to_csv(f'{NAME}_component_evaluation.csv', index=False)
    
    # Plot evaluation results
    plot_pca_evaluation(results_df)
    
    # Find the best number of components based on AUC
    best_auc_idx = results_df['f1'].idxmax()
    best_n_components = results_df.loc[best_auc_idx, 'n_components']
    
    # Alternative: Find best number of components based on explained variance (e.g., 95%)
    variance_threshold = 0.95
    variance_rows = results_df[results_df['explained_variance'] >= variance_threshold * 100]
    if not variance_rows.empty:
        variance_components = variance_rows.iloc[0]['n_components']
        print(f"Components needed for {variance_threshold*100}% variance: {variance_components}")
    else:
        variance_components = max(component_counts)
        print(f"No component count reaches {variance_threshold*100}% variance. Using maximum: {variance_components}")
    
    print(f"\nBest number of components by F1 score: {best_n_components}")
    
    # Use the best number of components
    final_n_components = int(best_n_components)
    print(f"\nUsing {final_n_components} components for final PCA transformation")
    
    # Apply the final PCA
    final_pca = PCA(n_components=final_n_components)
    X_train_pca = final_pca.fit_transform(X_train)
    X_val_pca = final_pca.transform(X_val)
    X_test_pca = final_pca.transform(X_test)
    
    # Plot correlation matrix for PCA components
    plot_correlation_matrix(X_train_pca, min(final_n_components, 20))  # Limit to 20 for readability
    
    # Create DataFrames with PCA-transformed data
    pca_columns = [f'PC{i+1}' for i in range(final_n_components)]
    
    train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
    train_pca_df['Produced an SEP'] = y_train
    train_pca_df = pd.concat([train_df[['Filename General', 'Most Probable AR Num', 'Blob Index']], train_pca_df], axis=1)
    
    val_pca_df = pd.DataFrame(X_val_pca, columns=pca_columns)
    val_pca_df['Produced an SEP'] = y_val
    val_pca_df = pd.concat([val_df[['Filename General', 'Most Probable AR Num', 'Blob Index']], val_pca_df], axis=1)
    
    test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)
    test_pca_df['Produced an SEP'] = y_test
    test_pca_df = pd.concat([test_df[['Filename General', 'Most Probable AR Num', 'Blob Index']], test_pca_df], axis=1)
    
    # Save PCA-transformed data
    train_pca_df.to_csv(f'{NAME}_train_pca.csv', index=False)
    val_pca_df.to_csv(f'{NAME}_val_pca.csv', index=False)
    test_pca_df.to_csv(f'{NAME}_test_pca.csv', index=False)
    
    # Save PCA model and metadata
    joblib.dump(final_pca, f'{NAME}_pca_model.joblib')
    joblib.dump({
        'selected_feature_indices': SELECTED_FEATURE_INDICES,
        'n_components': final_n_components,
        'pca_columns': pca_columns,
        'explained_variance_ratio': final_pca.explained_variance_ratio_,
        'cumulative_explained_variance': np.sum(final_pca.explained_variance_ratio_) * 100
    }, f'{NAME}_pca_metadata.joblib')
    
    # Save explained variance by component
    explained_variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(final_n_components)],
        'Explained_Variance_Ratio': final_pca.explained_variance_ratio_,
        'Cumulative_Explained_Variance': np.cumsum(final_pca.explained_variance_ratio_)
    })
    explained_variance_df.to_csv(f'{NAME}_explained_variance.csv', index=False)
    
    # Create an elbow plot for explained variance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, final_n_components + 1), final_pca.explained_variance_ratio_, marker='o')
    plt.title('Explained Variance by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{NAME}_explained_variance_ratio.png', dpi=300)
    
    # Also plot cumulative explained variance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, final_n_components + 1), np.cumsum(final_pca.explained_variance_ratio_), marker='o')
    plt.title('Cumulative Explained Variance by Principal Component')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{NAME}_cumulative_explained_variance.png', dpi=300)
    
    # Test a Random Forest model on the PCA-transformed data
    print('\nTraining Random Forest model on PCA-transformed data...')
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=cpus_to_use
    )
    
    rf_model.fit(X_train_pca, y_train)
    
    # Evaluate on validation set
    y_val_pred = rf_model.predict(X_val_pca)
    y_val_pred_proba = rf_model.predict_proba(X_val_pca)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    
    print(f'\nValidation results with {final_n_components} PCA components:')
    print(f'Accuracy: {val_accuracy:.4f}')
    print(f'Precision: {val_precision:.4f}')
    print(f'Recall: {val_recall:.4f}')
    print(f'AUC: {val_auc:.4f}')
    print(f'F1 Score: {val_f1:.4f}')
    
    # Evaluate on test set
    y_test_pred = rf_model.predict(X_test_pca)
    y_test_pred_proba = rf_model.predict_proba(X_test_pca)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    print(f'\nTest results with {final_n_components} PCA components:')
    print(f'Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall: {test_recall:.4f}')
    print(f'AUC: {test_auc:.4f}')
    print(f'F1 Score: {test_f1:.4f}')
    
    # Save the RF model
    joblib.dump(rf_model, f'{NAME}_rf_model.joblib')
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nPCA analysis completed in {total_time:.2f} seconds')

if __name__ == "__main__":
    main()