from NumericDataLoader import SEPInputDataGenerator
import pandas as pd
import numpy as np
import time
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

NAME = 'sep_prediction_feature_importance'

def build_feature_names():
    """
    Build a list of feature names based on the SEPInputDataGenerator's column definitions.
    This helps with interpreting feature importance later.
    
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

blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_all_events_including_new_flares_and_TEBBS_fix.csv'
blob_df = pd.read_csv(blob_df_filename)

print(f"Loaded dataset with {len(blob_df)} rows")

# Preprocess the data
blob_df['Produced an SEP'] = (blob_df['Number of SEPs Produced'] > 0) * 1  # 1 if produced, 0 otherwise
blob_df['Year'] = blob_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
blob_df['Is Plage'] = blob_df['Is Plage'].astype(int)
blob_df['Most Probable AR Num'] = blob_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])

# Split the data by year to maintain the same structure as in the NN script
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

# Print dataset statistics
print('\nDataset Statistics:')
print(f'Train set size: {len(train_df)}, SEP events: {train_df["Produced an SEP"].sum()} ({train_df["Produced an SEP"].mean()*100:.2f}%)')
print(f'Validation set size: {len(val_df)}, SEP events: {val_df["Produced an SEP"].sum()} ({val_df["Produced an SEP"].mean()*100:.2f}%)')
print(f'Test set size: {len(test_df)}, SEP events: {test_df["Produced an SEP"].sum()} ({test_df["Produced an SEP"].mean()*100:.2f}%)')

# NOTE: We don't do any oversampling/undersampling in this script

# Create data generators
batch_size = 32  # Can be larger since we'll extract all data at once
cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)

# Create the data generators
print('\nCreating data generators...')
train_generator = SEPInputDataGenerator(train_df, batch_size, False)
val_generator = SEPInputDataGenerator(val_df, batch_size, False)
test_generator = SEPInputDataGenerator(test_df, batch_size, False)

print(f'Number of batches in train: {len(train_generator)}')
print(f'Number of batches in val: {len(val_generator)}')
print(f'Number of batches in test: {len(test_generator)}')

# Function to extract all data from a generator
def extract_all_data(generator):
    all_X = []
    all_y = []
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        all_X.append(X_batch)
        all_y.append(y_batch)
    
    if len(all_X) > 0:
        return np.vstack(all_X), np.concatenate(all_y)
    else:
        return np.array([]), np.array([])

# Extract all data from generators
print('\nExtracting data from generators...')
X_train, y_train = extract_all_data(train_generator)
X_val, y_val = extract_all_data(val_generator)
X_test, y_test = extract_all_data(test_generator)

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

# Train Random Forest model
print('\nTraining Random Forest model...')
start_time = time.time()

# Calculate class weights to achieve 0.65 positive samples to 1 negative sample
class_weight = {0: 1, 1: 0.65}

# Initialize the model with custom class weights. TODO: Compare this to just using ROS/RUS
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=class_weight,
    random_state=42,
    n_jobs=cpus_to_use
)

rf_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f'Training completed in {training_time:.2f} seconds')

# Evaluate the model
print('\nEvaluating model...')
y_pred = rf_model.predict(X_val)
y_pred_proba = rf_model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

print(f'Validation Results:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC: {auc:.4f}')

# Display confusion matrix
cm = confusion_matrix(y_val, y_pred)
print(f'Confusion Matrix:')
print(cm)

# Feature Importance Analysis
print('\nAnalyzing feature importance...')
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(min(20, X_train.shape[1])):
    print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.6f})")

# Create a DataFrame with feature names and importance scores
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'StdDev': std
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Save feature importance to CSV
feature_importance_df.to_csv(f'{NAME}_feature_importance.csv', index=False)
print(f'Feature importance saved to {NAME}_feature_importance.csv')

# Plot feature importance (top 30 features)
plt.figure(figsize=(12, 10))
top_n = 30
plt.barh(feature_importance_df['Feature'][:top_n], 
        feature_importance_df['Importance'][:top_n], 
        xerr=feature_importance_df['StdDev'][:top_n],
        color='skyblue')
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n} Features by Importance')
plt.tight_layout()
plt.savefig(f'{NAME}_top_{top_n}_features.png', dpi=300)
print(f'Feature importance plot saved to {NAME}_top_{top_n}_features.png')

# Create a correlation matrix of the top features
top_features = feature_importance_df['Feature'][:20].tolist()
all_features_df = pd.DataFrame(X_train, columns=feature_names)
top_features_df = all_features_df[top_features]

plt.figure(figsize=(14, 12))
correlation_matrix = top_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Top 20 Features')
plt.tight_layout()
plt.savefig(f'{NAME}_top_features_correlation.png', dpi=300)
print(f'Correlation matrix saved to {NAME}_top_features_correlation.png')

# Group features by category
print('\nFeature Category Analysis:')
feature_categories = {
    'one_time': [],
    'current_timestep': [],
    'previous_timesteps': []
}

one_time_info_count = len(SEPInputDataGenerator.BLOB_ONE_TIME_INFO)
vector_cols_count = len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL)

for i, name in enumerate(feature_names):
    if i < one_time_info_count:
        feature_categories['one_time'].append(i)
    elif i < one_time_info_count + vector_cols_count:
        feature_categories['current_timestep'].append(i)
    else:
        feature_categories['previous_timesteps'].append(i)

# Calculate importance by category
category_importance = {}
for category, indices in feature_categories.items():
    category_importance[category] = importances[indices].sum()

print('Importance by Feature Category:')
for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
    print(f'{category}: {importance:.6f} ({importance*100:.2f}%)')

# Calculate cumulative importance and find threshold for feature selection
feature_importance_df['CumulativeImportance'] = feature_importance_df['Importance'].cumsum()

# Find features needed for different importance thresholds
thresholds = [0.75, 0.85, 0.95, 0.99]
features_at_threshold = {}

for threshold in thresholds:
    features = feature_importance_df[feature_importance_df['CumulativeImportance'] <= threshold]
    
    if len(features) == 0:
        features = feature_importance_df.iloc[:1]
    
    while features['CumulativeImportance'].iloc[-1] < threshold:
        next_idx = len(features)
        if next_idx < len(feature_importance_df):
            features = pd.concat([features, feature_importance_df.iloc[next_idx:next_idx+1]])
        else:
            break
    
    features_at_threshold[threshold] = features

# Print feature recommendations for different thresholds
print('\nFeature Selection Recommendations:')
for threshold, features in features_at_threshold.items():
    num_features = len(features)
    cumulative_importance = features['CumulativeImportance'].iloc[-1]
    print(f'For {threshold*100:.0f}% importance threshold: {num_features} features ({cumulative_importance*100:.2f}%)')

# Recommend the 95% threshold as default
recommended_features = features_at_threshold[0.95]
recommended_features.to_csv(f'{NAME}_recommended_features_95percent.csv', index=False)
print(f'\nRecommended features (95% importance threshold) saved to {NAME}_recommended_features_95percent.csv')

# Test the model on the test set
print('\nFinal evaluation on test set:')
y_test_pred = rf_model.predict(X_test)
y_test_pred_proba = rf_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test AUC: {test_auc:.4f}')

# Display confusion matrix for test set
cm_test = confusion_matrix(y_test, y_test_pred)
print(f'Test Confusion Matrix:')
print(cm_test)

# Save the model
joblib.dump(rf_model, f'{NAME}_model.joblib')
print(f'Model saved to {NAME}_model.joblib')

# Save the feature names and scaler for future use
joblib.dump({
    'feature_names': feature_names,
    'scaler': scaler,
    'recommended_features': recommended_features['Feature'].tolist()
}, f'{NAME}_metadata.joblib')
print(f'Model metadata saved to {NAME}_metadata.joblib')

print('\nFeature importance analysis complete!')
