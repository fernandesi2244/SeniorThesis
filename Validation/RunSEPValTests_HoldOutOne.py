"""
Run the SEPVal tests on the optimal models for photospheric, coronal, and combination data,
using a hold-out-one strategy.

Negative events list: SEPValidationChallengePhaseIII_NonEvents_v4 - SEPValidationChallengePhaseIII_NonEvents_v4.csv
Positive events list: SEPValidationChallengePhaseIII_SEPevents_v4 - SEPValidationChallengePhaseIII_SEPevents_v4.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import TrainSEPValModel_HoldOutOne as TrainSEPValModel

# Define data subsets to test
data_subsets = ['photospheric', 'coronal', 'numeric']

def main():
    start_time = time.time()
    print(f"Starting Hold-Out-One testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Open the CSV files containing the positive and negative events
    non_events = pd.read_csv('../InputData/SEPValidationChallengePhaseIII_NonEvents_v4.csv')
    sep_events = pd.read_csv('../InputData/SEPValidationChallengePhaseIII_SEPevents_v4.csv')
    
    # Extract active region numbers from events
    non_event_ranges = []
    non_event_ars = []
    
    event_ranges = []
    event_ars = []
    
    # Iterate through each non_event and mark for removal later
    for index, row in non_events.iterrows():
        forecast_start_time = row['Event Period Start Time for Continuous Forecast Models (24 hours prior to flare)']
        # Parse the date string into a datetime object (month/day/year hour:minute)
        forecast_start_time = pd.to_datetime(forecast_start_time, format='%m/%d/%Y %H:%M')
    
        forecast_end_time = row['Event Period End Time for Continuous Forecast Models (~14 hours after flare)']
        forecast_end_time = pd.to_datetime(forecast_end_time, format='%m/%d/%Y %H:%M')
    
        ar_num = row['Active Region']
        try:
            ar_num = int(ar_num)
        except ValueError:
            ar_num = None
    
        non_event_ranges.append((forecast_start_time, forecast_end_time))
        non_event_ars.append(ar_num)
    
    # Iterate through each sep_event and mark for removal later
    for index, row in sep_events.iterrows():
        forecast_start_time = row['Event Period Start Time for Continuous Forecast Models (24 hours prior to start)']
        # Parse the date string into a datetime object (month/day/year hour:minute)
        forecast_start_time = pd.to_datetime(forecast_start_time, format='%m/%d/%Y %H:%M')
        
        forecast_end_time = row['Event Period End Time for Continuous Forecast Models (to peak - can go past if makes sense for your model)']
        forecast_end_time = pd.to_datetime(forecast_end_time, format='%m/%d/%Y %H:%M')
    
        ar_num = row['AR']
        try:
            ar_num = int(ar_num)
        except ValueError:
            ar_num = None
    
        event_ranges.append((forecast_start_time, forecast_end_time))
        event_ars.append(ar_num)
    
    print('Number of non-None ARs in non-events:', len([ar for ar in non_event_ars if ar is not None]))
    print('Number of non-None ARs in events:', len([ar for ar in event_ars if ar is not None]))

    # Create directories for results
    os.makedirs('results', exist_ok=True)
    
    # For each data subset
    for data_subset in data_subsets:
        print(f"\n{'='*80}")
        print(f"Processing {data_subset} data subset")
        print(f"{'='*80}")
        
        # Load the unified data
        unified_data = pd.read_csv('../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv')
        
        # Convert datetime string to datetime object
        unified_data['datetime_dt'] = pd.to_datetime(unified_data['datetime'], format='%Y%m%d_%H%M%S_TAI')
        unified_data['Most Probable AR Num'] = unified_data['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])
        
        # Create a directory for this data subset's results
        subset_results_dir = f'results/{data_subset}_holdout1'
        os.makedirs(subset_results_dir, exist_ok=True)
        
        # Initialize aggregated results
        all_true_labels = []
        all_predictions = []
        all_probabilities = []
        all_test_indices = []
        
        # Process non-events
        print("\nProcessing Non-Events:")
        for i, (start_time, end_time) in enumerate(non_event_ranges):
            process_event(i, non_event_ars[i], start_time, end_time, 'non_event', unified_data, data_subset, subset_results_dir, 
                         all_true_labels, all_predictions, all_probabilities, all_test_indices)
            
        # Process SEP events
        print("\nProcessing SEP Events:")
        for i, (start_time, end_time) in enumerate(event_ranges):
            process_event(i, event_ars[i], start_time, end_time, 'sep_event', unified_data, data_subset, subset_results_dir,
                         all_true_labels, all_predictions, all_probabilities, all_test_indices)
        
        # Calculate and display aggregated results
        print("\nCalculating aggregated results...")
        
        # Convert lists to numpy arrays
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save the confusion matrix as CSV
        np.savetxt(f'{subset_results_dir}/confusion_matrix.csv', cm, delimiter=',', fmt='%d')
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No SEP', 'SEP'], yticklabels=['No SEP', 'SEP'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {data_subset.capitalize()} Data')
        plt.tight_layout()
        plt.savefig(f'{subset_results_dir}/confusion_matrix.png')
        
        # Calculate metrics
        calculate_and_save_metrics(y_true, y_pred, y_proba, subset_results_dir)
        
        # Save all individual predictions
        prediction_df = pd.DataFrame({
            'test_index': all_test_indices,
            'true_label': all_true_labels,
            'prediction': all_predictions,
            'probability': all_probabilities
        })
        prediction_df.to_csv(f'{subset_results_dir}/all_predictions.csv', index=False)
        
        print(f"\nResults for {data_subset} saved to {subset_results_dir}")
        print(f"Confusion Matrix for {data_subset} data subset:")
        print(cm)
        print("\n")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f'\nHold-Out-One testing completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)')

def process_event(event_idx, ar_num, start_time, end_time, event_type, unified_data, data_subset, results_dir, 
                 all_true_labels, all_predictions, all_probabilities, all_test_indices):
    """Process a single event (non-event or SEP event)"""
    
    # Set expected label based on event type
    expected_label = 0 if event_type == 'non_event' else 1

    # Find data points within the event time range (plus 2 days)
    time_filter = ((unified_data['datetime_dt'] >= start_time) &
                  (unified_data['datetime_dt'] <= end_time + pd.Timedelta(days=2)))
    
    test_data_time = unified_data[time_filter].copy()
    
    # Also add points with matching AR number if available
    ar_data = pd.DataFrame()
    if ar_num is not None:
        ar_filter = (unified_data['Most Probable AR Num'] == str(ar_num))
        ar_data = unified_data[ar_filter].copy()
    
    # Combine time-based and AR-based test data
    test_data = pd.concat([test_data_time, ar_data]).drop_duplicates()
    
    if len(test_data) == 0:
        print(f"No test data found for {event_type} {event_idx+1}")
        return
    
    print(f"Processing {event_type} {event_idx+1}: found {len(test_data)} test points")
    
    # Process each individual test point
    for test_idx, test_row in test_data.iterrows():
        print(f"  Test point {test_idx} ({test_row['datetime']})")
        
        # Create training set (all data except this test point)
        train_data = unified_data[unified_data.index != test_idx].copy()
        
        # Test on the single point
        single_test_data = unified_data.iloc[[test_idx]].copy()
        
        # Train model and get prediction
        prediction, probability = TrainSEPValModel.main(
            data_subset, 
            train_data, 
            single_test_data,
            f"{results_dir}/point_{test_idx}"
        )
        
        # Store results
        all_true_labels.append(expected_label)
        all_predictions.append(prediction)
        all_probabilities.append(probability)
        all_test_indices.append(test_idx)
        
        # Save individual result
        with open(f"{results_dir}/point_{test_idx}_result.txt", "w") as f:
            f.write(f"Test Index: {test_idx}\n")
            f.write(f"Datetime: {test_row['datetime']}\n")
            f.write(f"Event Type: {event_type}\n")
            f.write(f"Expected Label: {expected_label}\n")
            f.write(f"Predicted Label: {prediction}\n")
            f.write(f"Probability: {probability:.4f}\n")

def calculate_and_save_metrics(y_true, y_pred, y_proba, results_dir):
    """Calculate and save performance metrics"""
    
    # Extract components from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f_half = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    # Calculate HSS and TSS
    hss = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)) if ((tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)) > 0 else 0
    tss = (tp / (tp + fn) - fp / (fp + tn)) if ((tp + fn) > 0 and (fp + tn) > 0) else 0
    
    # Calculate AUC if possible (need both positive and negative examples)
    auc = 0
    try:
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_proba)
    except:
        pass
    
    # Display metrics
    print(f"\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F0.5 Score: {f_half:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"HSS: {hss:.4f}")
    print(f"TSS: {tss:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Save metrics to file
    with open(f"{results_dir}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"F0.5 Score: {f_half:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"HSS: {hss:.4f}\n")
        f.write(f"TSS: {tss:.4f}\n")
        f.write(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

if __name__ == "__main__":
    main()
