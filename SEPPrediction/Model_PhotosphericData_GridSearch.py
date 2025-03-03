from PhotosphericDataLoader import SEPInputDataGenerator
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

NAME = 'sep_prediction_photospheric_data_grid_search'

# Create output directory for results
RESULTS_DIR = f'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Multiprocessing setup
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

def build_feature_names():
    """
    Build a list of feature names based on the SEPInputDataGenerator's column definitions.
    
    Returns:
        List of feature names
    """
    # One-time features
    feature_names = list(SEPInputDataGenerator.BLOB_ONE_TIME_INFO)
    
    # Time-series features
    for t in range(SEPInputDataGenerator.TIMESERIES_STEPS):
        for col in SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL:
            if t == 0:
                feature_names.append(f"{col}")
            else:
                feature_names.append(f"{col}_t-{t*4}")
    
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
    # Create pivot table for heatmap
    pivot_df = results_df.pivot(index='n_components', columns='n_features', values=metric)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.4f')
    plt.title(f'{metric.upper()} Score by Feature Count and PCA Components')
    plt.xlabel('Number of Features')
    plt.ylabel('Number of PCA Components')
    plt.savefig(f'{RESULTS_DIR}/{NAME}_{metric}_heatmap.png', dpi=300)
    
    # For each feature count, plot metrics vs number of components
    plt.figure(figsize=(15, 10))
    feature_counts = results_df['n_features'].unique()
    
    for feat_count in feature_counts:
        subset = results_df[results_df['n_features'] == feat_count]
        plt.plot(subset['n_components'], subset[metric], marker='o', label=f'{feat_count} features')
    
    plt.title(f'{metric.upper()} Score vs PCA Components')
    plt.xlabel('Number of PCA Components')
    plt.ylabel(f'{metric.upper()} Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{RESULTS_DIR}/{NAME}_{metric}_by_components.png', dpi=300)

def main():
    """Main function to run the combined feature selection and PCA analysis"""
    start_time = time.time()
    print(f"Starting combined feature selection and PCA analysis at {time.ctime()}")

    train_features_file = f'{NAME}_X_train_data.npy'
    train_labels_file = f'{NAME}_y_train_data.npy'

    val_features_file = f'{NAME}_val_data.npy'
    val_labels_file = f'{NAME}_val_labels.npy'

    test_features_file = f'{NAME}_test_data.npy'
    test_labels_file = f'{NAME}_test_labels.npy'

    if os.path.exists(test_labels_file):
        # Load in the npy files as numpy arrays
        X_train = np.load(train_features_file)
        y_train = np.load(train_labels_file)

        X_val = np.load(val_features_file)
        y_val = np.load(val_labels_file)

        X_test = np.load(test_features_file)
        y_test = np.load(test_labels_file)
    else:
        # Load dataset
        blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_all_events_including_new_flares_and_TEBBS_fix.csv'
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
        
        # Standardize the features
        scaler = StandardScaler()
        cols_to_scale = SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL + SEPInputDataGenerator.BLOB_ONE_TIME_INFO
        train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
        val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
        test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
        
        # Apply class balancing to training set
        print('\nBefore resampling:')
        print('Train set count:', len(train_df))
        print('Train set SEP count:', train_df['Produced an SEP'].sum())
        
        # Over-sample minority class
        ros = RandomOverSampler(sampling_strategy=0.325, random_state=42)
        train_df, _ = ros.fit_resample(train_df, train_df['Produced an SEP'])
        
        # Under-sample majority class
        rus = RandomUnderSampler(sampling_strategy=0.65, random_state=42)
        train_df, _ = rus.fit_resample(train_df, train_df['Produced an SEP'])
        
        print('After resampling:')
        print('Train set count:', len(train_df))
        print('Train set SEP count:', train_df['Produced an SEP'].sum())
        
        # Print dataset statistics
        print('\nDataset Statistics:')
        print(f'Train set size: {len(train_df)}, SEP events: {train_df["Produced an SEP"].sum()} ({train_df["Produced an SEP"].mean()*100:.2f}%)')
        print(f'Validation set size: {len(val_df)}, SEP events: {val_df["Produced an SEP"].sum()} ({val_df["Produced an SEP"].mean()*100:.2f}%)')
        print(f'Test set size: {len(test_df)}, SEP events: {test_df["Produced an SEP"].sum()} ({test_df["Produced an SEP"].mean()*100:.2f}%)')
        
        # Create the data generators (right now, just being used to get timeseries data)
        print('\nCreating data generators...')
        batch_size = 32
        train_generator = SEPInputDataGenerator(train_df, batch_size, False, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
        val_generator = SEPInputDataGenerator(val_df, batch_size, False, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
        test_generator = SEPInputDataGenerator(test_df, batch_size, False, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
        
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

    # Build feature names for interpretation
    feature_names = build_feature_names()
    
    print(f'Data extraction complete.')
    print(f'Train data shape: {X_train.shape}, labels shape: {y_train.shape}')
    print(f'Validation data shape: {X_val.shape}, labels shape: {y_val.shape}')
    print(f'Test data shape: {X_test.shape}, labels shape: {y_test.shape}')
    print(f'Number of features: {X_train.shape[1]}')
    
    # Check for NaN and Inf values
    print('\nChecking for NaN and Inf values:')
    print(f'Train NaN count: {np.isnan(X_train).sum()}, Inf count: {np.isinf(X_train).sum()}')
    print(f'Val NaN count: {np.isnan(X_val).sum()}, Inf count: {np.isinf(X_val).sum()}')
    print(f'Test NaN count: {np.isnan(X_test).sum()}, Inf count: {np.isinf(X_test).sum()}')
    
    # Replace any NaN or Inf values with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Run feature selection
    feature_indices = feature_selection(X_train, y_train, feature_names)
    
    # Define feature counts to test
    feature_counts = [20, 40, 60, 80, 100]
    
    # Define component counts to test for PCA
    component_counts = [2, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    
    # Create a list to store all results
    all_results = []
    best_f1 = 0
    best_config = None
    best_model = None
    best_pca = None
    
    # Loop through different feature counts
    for n_features in feature_counts:
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
            print('\n' + '-'*50)
            print(f'Feature Count: {n_features}, PCA Components: {n_components}')
            print('-'*50)
            
            # Evaluate this configuration
            result = evaluate_pca_component(
                X_train_selected, y_train, 
                X_val_selected, y_val, 
                n_features, n_components
            )
            
            # Extract metrics to store in results list
            result_row = {
                'n_features': n_features,
                'n_components': n_components,
                'explained_variance': result['explained_variance'],
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1': result['metrics']['f1'],
                'auc': result['metrics']['auc']
            }
            
            all_results.append(result_row)
            
            # Check if this is the best configuration based on F1 score
            if result['metrics']['f1'] > best_f1:
                best_f1 = result['metrics']['f1']
                best_config = {
                    'n_features': n_features,
                    'n_components': n_components,
                    'feature_indices': selected_indices,
                    'metrics': result['metrics']
                }
                best_model = result['model']
                best_pca = result['pca']
                
                print(f"New best configuration found: {n_features} features, {n_components} PCA components, F1: {best_f1:.4f}")
    
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
    
    # Print best configuration
    print("\nBest Configuration:")
    print(f"Number of Features: {best_config['n_features']}")
    print(f"Number of PCA Components: {best_config['n_components']}")
    print(f"F1 Score: {best_config['metrics']['f1']:.4f}")
    print(f"Accuracy: {best_config['metrics']['accuracy']:.4f}")
    print(f"Precision: {best_config['metrics']['precision']:.4f}")
    print(f"Recall: {best_config['metrics']['recall']:.4f}")
    print(f"AUC: {best_config['metrics']['auc']:.4f}")
    
    # Save best feature indices
    best_feature_names = [feature_names[i] for i in best_config['feature_indices']]
    pd.DataFrame({'Feature': best_feature_names}).to_csv(f'{RESULTS_DIR}/{NAME}_best_features.csv', index=False)
    
    # Now, evaluate the best model on the test set
    print("\nEvaluating best model on test set...")
    
    # Get the best configuration's selected features
    X_test_selected = X_test[:, best_config['feature_indices']]
    
    # Apply PCA transformation
    X_test_pca = best_pca.transform(X_test_selected)
    
    # Make predictions
    y_test_pred = best_model.predict(X_test_pca)
    y_test_pred_proba = best_model.predict_proba(X_test_pca)[:, 1]
    
    # Evaluate performance
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Save the best model, PCA transformer, and feature indices
    model_metadata = {
        'feature_indices': best_config['feature_indices'],
        'feature_names': best_feature_names,
        'n_components': best_config['n_components'],
        'train_metrics': best_config['metrics'],
        'test_metrics': test_metrics,
        'scaler': scaler
    }
    
    # Save model and metadata
    joblib.dump(best_model, f'{RESULTS_DIR}/{NAME}_best_model.joblib')
    joblib.dump(best_pca, f'{RESULTS_DIR}/{NAME}_best_pca.joblib')
    joblib.dump(model_metadata, f'{RESULTS_DIR}/{NAME}_metadata.joblib')
    
    print(f"\nBest model and metadata saved to {RESULTS_DIR}/ directory")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nAnalysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')

if __name__ == "__main__":
    main()