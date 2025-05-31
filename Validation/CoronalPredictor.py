"""
Coronal SEP Prediction Script

This script loads the trained coronal model and makes predictions on input feature vectors,
outputting results in CCMC JSON format. Enhanced to compare predictions from both final
models and results models.
"""

import numpy as np
import pandas as pd
import joblib
import json
import datetime
from datetime import timedelta
import os
import sys
import pathlib
import glob

# Add the SEPPrediction directory to the path
rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir, 'SEPPrediction'))

class CoronalPredictor:
    def __init__(self, final_model_path='final_models/coronal_final_model.joblib', 
                 results_model_dir='../SEPPrediction/results/coronal_data/'):
        """
        Initialize the coronal predictor
        
        Args:
            final_model_path: Path to the final trained model file
            results_model_dir: Directory containing the results model file
        """
        self.final_model_path = final_model_path
        self.results_model_dir = results_model_dir
        
        # Final model attributes
        self.final_model_data = None
        self.final_model = None
        self.final_scaler = None
        self.final_pca = None
        self.final_feature_indices = None
        self.final_feature_names = None
        
        # Results model attributes
        self.results_model_data = None
        self.results_model = None
        self.results_scaler = None
        self.results_pca = None
        self.results_feature_indices = None
        self.results_feature_names = None
        self.results_model_path = None
        
        self.load_models()
    
    def find_results_model(self):
        """Find the joblib file in the results directory"""
        if not os.path.exists(self.results_model_dir):
            raise FileNotFoundError(f"Results model directory not found: {self.results_model_dir}")
        
        joblib_files = glob.glob(os.path.join(self.results_model_dir, "*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(f"No joblib files found in: {self.results_model_dir}")
        
        if len(joblib_files) > 1:
            print(f"Warning: Multiple joblib files found, using first one: {joblib_files[0]}")
        
        return joblib_files[0]
    
    def load_models(self):
        """Load both the final and results models"""
        # Load final model
        if not os.path.exists(self.final_model_path):
            raise FileNotFoundError(f"Final model file not found: {self.final_model_path}")
        
        print(f"Loading final coronal model from {self.final_model_path}")
        self.final_model_data = joblib.load(self.final_model_path)
        
        self.final_model = self.final_model_data['model']
        self.final_scaler = self.final_model_data['scaler']
        self.final_pca = self.final_model_data.get('pca', None)
        self.final_feature_indices = self.final_model_data['feature_indices']
        self.final_feature_names = self.final_model_data['feature_names']
        
        print(f"Final model loaded successfully")
        print(f"Final model features: {len(self.final_feature_names)}")
        print(f"Final model PCA applied: {'Yes' if self.final_pca is not None else 'No'}")
        
        # Load results model
        self.results_model_path = self.find_results_model()
        print(f"\nLoading results coronal model from {self.results_model_path}")
        self.results_model_data = joblib.load(self.results_model_path)
        
        self.results_model = self.results_model_data['model']
        self.results_scaler = self.results_model_data['scaler']
        self.results_pca = self.results_model_data.get('pca', None)
        self.results_feature_indices = self.results_model_data['feature_indices']
        self.results_feature_names = self.results_model_data['feature_names']
        
        print(f"Results model loaded successfully")
        print(f"Results model features: {len(self.results_feature_names)}")
        print(f"Results model PCA applied: {'Yes' if self.results_pca is not None else 'No'}")
    
    def preprocess_features(self, feature_vector, model_type='final'):
        """
        Preprocess the input feature vector
        
        Args:
            feature_vector: Input feature vector (numpy array or list)
            model_type: 'final' or 'results' to specify which model's preprocessing to use
            
        Returns:
            Preprocessed feature vector ready for prediction
        """
        # Select the appropriate preprocessing components
        if model_type == 'final':
            scaler = self.final_scaler
            pca = self.final_pca
            feature_indices = self.final_feature_indices
        else:
            scaler = self.results_scaler
            pca = self.results_pca
            feature_indices = self.results_feature_indices
        
        # Convert to numpy array if needed
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector)
        
        # Ensure it's 2D for sklearn
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Handle NaN and Inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply scaling
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Select features used during training
        feature_vector_selected = feature_vector_scaled[:, feature_indices]
        
        # Apply PCA if it was used during training
        if pca is not None:
            feature_vector_final = pca.transform(feature_vector_selected)
        else:
            feature_vector_final = feature_vector_selected
        
        return feature_vector_final
    
    def predict(self, feature_vector):
        """
        Make prediction on input feature vector using both models
        
        Args:
            feature_vector: Input feature vector
            
        Returns:
            Dictionary containing predictions and probabilities from both models
        """
        # Preprocess the feature vector for both models
        X_final = self.preprocess_features(feature_vector, 'final')
        X_results = self.preprocess_features(feature_vector, 'results')
        
        # Make predictions with final model
        final_prediction = self.final_model.predict(X_final)[0]
        final_prediction_proba = self.final_model.predict_proba(X_final)[0][1]
        
        # Make predictions with results model
        results_prediction = self.results_model.predict(X_results)[0]
        results_prediction_proba = self.results_model.predict_proba(X_results)[0][1]
        
        return {
            'final_model': {
                'prediction': int(final_prediction),
                'probability': float(final_prediction_proba)
            },
            'results_model': {
                'prediction': int(results_prediction),
                'probability': float(results_prediction_proba)
            }
        }
    
    def create_ccmc_json(self, input_dt, prediction_result, model_type='final', output_path=None, issueTime=None):
        """
        Create CCMC JSON format output
        
        Args:
            input_dt: Input datetime (datetime object or string)
            prediction_result: Dictionary from predict() method for specific model
            model_type: 'final' or 'results' to specify which model result to use
            output_path: Path to save JSON file (optional)
            issueTime: Issue time (optional, defaults to current UTC time)
            
        Returns:
            Dictionary in CCMC JSON format
        """
        # Parse datetime if it's a string
        if isinstance(input_dt, str):
            # Try different datetime formats
            try:
                input_dt = pd.to_datetime(input_dt, format='%Y%m%d_%H%M%S_TAI')
            except:
                try:
                    input_dt = pd.to_datetime(input_dt)
                except:
                    raise ValueError(f"Could not parse datetime: {input_dt}")
        
        # Extract datetime components
        year = input_dt.year
        month = input_dt.month
        day = input_dt.day
        hour = input_dt.hour
        minute = input_dt.minute
        
        dt_str = datetime.datetime.strftime(input_dt, '%Y-%m-%dT%H:%M:%S')
        
        # Calculate forecast end time (24 hours later)
        forecastEndTime = input_dt + timedelta(days=1)
        forecastEndTimeStr = datetime.datetime.strftime(forecastEndTime, '%Y-%m-%dT%H:%M:%S')
        
        # Set issue time
        if issueTime is None:
            issueTime = datetime.datetime.now(datetime.timezone.utc)
        issueTimeStr = datetime.datetime.strftime(issueTime, '%Y-%m-%dT%H:%M:%S')
        
        # Create CCMC JSON structure
        model_suffix = f"_{model_type}" if model_type == 'results' else ""
        ccmc_json = {
            "sep_forecast_submission": {
                "model": {
                    "short_name": f"MagPy_ML_SHARP_HMI_CEA_coronal{model_suffix}",
                    "spase_id": f"spase://CCMC/SimulationModel/MagPy-ML/coronal{model_suffix}/v1"
                },
                "mode": "forecast",
                "issue_time": f"{issueTimeStr}Z",
                "inputs": [
                    {
                        "magnetogram": {
                            "observatory": "SDO",
                            "instrument": "HMI",
                            "products": [
                                {
                                    "product": "hmi.sharp_cea_720s_nrt",
                                    "last_data_time": f'{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}Z',
                                }
                            ]
                        }
                    }
                ],
                "forecasts": [
                    {
                        "energy_channel": {
                            "min": 10,
                            "max": -1,
                            "units": "MeV"
                        },
                        "species": "proton",
                        "location": "earth",
                        "prediction_window": {
                            "start_time": f"{dt_str}Z",
                            "end_time": f"{forecastEndTimeStr}Z",
                        },
                        "probabilities": [
                            {
                                "probability_value": f"{prediction_result['probability']:.5f}",
                                "threshold": 10,
                                "threshold_units": "pfu"
                            }
                        ],
                        "all_clear": {
                            "all_clear_boolean": not bool(prediction_result['prediction']),
                            "threshold": 10,
                            "threshold_units": "pfu",
                            "probability_threshold": 0.5
                        }
                    }
                ]
            }
        }
        
        # Save to file if output path is provided
        if output_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf8') as json_file:
                json.dump(ccmc_json, json_file, indent=2, separators=(',', ': '))
            
            print(f"CCMC JSON saved to: {output_path}")
        
        return ccmc_json
    
    def predict_and_export(self, feature_vector, input_dt, output_dir='predictions/coronal', issueTime=None):
        """
        Make prediction and export CCMC JSON for both models
        
        Args:
            feature_vector: Input feature vector
            input_dt: Input datetime
            output_dir: Directory to save JSON files
            issueTime: Issue time (optional)
            
        Returns:
            Dictionary containing prediction results and file paths for both models
        """
        # Make predictions with both models
        prediction_results = self.predict(feature_vector)
        
        # Parse datetime for filename
        if isinstance(input_dt, str):
            try:
                dt_parsed = pd.to_datetime(input_dt, format='%Y%m%d_%H%M%S_TAI')
            except:
                dt_parsed = pd.to_datetime(input_dt)
        else:
            dt_parsed = input_dt
        
        # Set issue time
        if issueTime is None:
            issueTime = datetime.datetime.now(datetime.timezone.utc)
        
        # Create filenames for both models
        base_filename = f'MagPy-ML-HMI-SHARP-Vector-coronal.{dt_parsed.year:04d}{dt_parsed.month:02d}{dt_parsed.day:02d}T{dt_parsed.hour:02d}{dt_parsed.minute:02d}.{issueTime.year:04d}{issueTime.month:02d}{issueTime.day:02d}T{issueTime.hour:02d}{issueTime.minute:02d}'
        
        final_filename = f'{base_filename}_final.json'
        results_filename = f'{base_filename}_results.json'
        
        final_output_path = os.path.join(output_dir, final_filename)
        results_output_path = os.path.join(output_dir, results_filename)
        
        # Create CCMC JSONs for both models
        final_ccmc_json = self.create_ccmc_json(input_dt, prediction_results['final_model'], 'final', final_output_path, issueTime)
        results_ccmc_json = self.create_ccmc_json(input_dt, prediction_results['results_model'], 'results', results_output_path, issueTime)
        
        return {
            'final_model': {
                'prediction': prediction_results['final_model']['prediction'],
                'probability': prediction_results['final_model']['probability'],
                'json_file': final_output_path,
                'ccmc_json': final_ccmc_json
            },
            'results_model': {
                'prediction': prediction_results['results_model']['prediction'],
                'probability': prediction_results['results_model']['probability'],
                'json_file': results_output_path,
                'ccmc_json': results_ccmc_json
            },
            'comparison': {
                'prediction_agreement': prediction_results['final_model']['prediction'] == prediction_results['results_model']['prediction'],
                'probability_difference': abs(prediction_results['final_model']['probability'] - prediction_results['results_model']['probability'])
            }
        }

def main():
    """
    Example usage of the CoronalPredictor
    """
    try:
        # Initialize predictor
        predictor = CoronalPredictor()
        
        # ============================================================================
        # USER INPUT SECTION - MODIFY THIS SECTION WITH YOUR ACTUAL DATA
        # ============================================================================
        
        # Coronal feature vector components (based on CoronalDataLoader.py):
        # 
        # One-time info (11 features):
        # ['Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 
        #  'Max Product of Half Angle and Speed of Recent CMEs', 'Number of Sunspots', 
        #  'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 
        #  'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares', 
        #  'Number of Recent SEPs', 'Number of Recent Subthreshold SEPs']
        
        # Per-blob vector columns (14 features per blob per timestep):
        # ['Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 
        #  'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 
        #  'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 
        #  'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 
        #  'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 
        #  'Volume Total Unsigned Magnetic Flux', 'Is Plage', 'Stonyhurst Longitude']
        
        # For per-disk-4hr granularity: 11 + (5 blobs × 6 timesteps × 14 features) = 11 + 420 = 431 features
        
        # REPLACE THIS WITH YOUR ACTUAL FEATURE VECTOR:
        your_feature_vector = [
            # One-time info (11 values) - REPLACE WITH YOUR DATA
            0.0,  # Number of Recent Flares
            0.0,  # Max Class Type of Recent Flares
            0.0,  # Number of Recent CMEs
            0.0,  # Max Product of Half Angle and Speed of Recent CMEs
            0.0,  # Number of Sunspots
            0.0,  # Max Flare Peak of Recent Flares
            0.0,  # Min Temperature of Recent Flares
            0.0,  # Median Emission Measure of Recent Flares
            0.0,  # Median Duration of Recent Flares
            0.0,  # Number of Recent SEPs
            0.0,  # Number of Recent Subthreshold SEPs
            
            # Blob data for 5 blobs × 6 timesteps × 14 features (420 values total)
            # REPLACE ALL THESE ZEROS WITH YOUR ACTUAL DATA
            *([0.0] * 420)  # 420 zeros as placeholder - replace with your blob timeseries data
        ]
        
        # Your input datetime - REPLACE WITH YOUR ACTUAL DATETIME
        your_datetime = datetime.datetime(2023, 6, 15, 12, 0, 0)  # MODIFY THIS
        
        # ============================================================================
        # END USER INPUT SECTION
        # ============================================================================
        
        print(f"Feature vector length: {len(your_feature_vector)}")
        print("Expected length for coronal per-disk-4hr: 431")
        
        if len(your_feature_vector) != 431:
            print(f"WARNING: Feature vector length ({len(your_feature_vector)}) doesn't match expected length (431)")
            print("Please check your feature vector dimensions")
        
        # Make prediction and export for both models
        result = predictor.predict_and_export(
            feature_vector=your_feature_vector,
            input_dt=your_datetime,
            output_dir='predictions/coronal'
        )
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS COMPARISON")
        print("="*60)
        
        print(f"\nFinal Model:")
        print(f"  Prediction: {result['final_model']['prediction']}")
        print(f"  Probability: {result['final_model']['probability']:.4f}")
        print(f"  JSON file: {result['final_model']['json_file']}")
        
        print(f"\nResults Model:")
        print(f"  Prediction: {result['results_model']['prediction']}")
        print(f"  Probability: {result['results_model']['probability']:.4f}")
        print(f"  JSON file: {result['results_model']['json_file']}")
        
        print(f"\nComparison:")
        print(f"  Predictions agree: {result['comparison']['prediction_agreement']}")
        print(f"  Probability difference: {result['comparison']['probability_difference']:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()