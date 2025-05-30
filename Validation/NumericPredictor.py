"""
Numeric SEP Prediction Script

This script loads the trained numeric model and makes predictions on input feature vectors,
outputting results in CCMC JSON format.
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

# Add the SEPPrediction directory to the path
rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir, 'SEPPrediction'))

class NumericPredictor:
    def __init__(self, model_path='final_models/numeric_final_model.joblib'):
        """
        Initialize the numeric predictor
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_indices = None
        self.feature_names = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing artifacts"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading numeric model from {self.model_path}")
        self.model_data = joblib.load(self.model_path)
        
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.pca = self.model_data.get('pca', None)
        self.feature_indices = self.model_data['feature_indices']
        self.feature_names = self.model_data['feature_names']
        
        print(f"Model loaded successfully")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"PCA applied: {'Yes' if self.pca is not None else 'No'}")
    
    def preprocess_features(self, feature_vector):
        """
        Preprocess the input feature vector
        
        Args:
            feature_vector: Input feature vector (numpy array or list)
            
        Returns:
            Preprocessed feature vector ready for prediction
        """
        # Convert to numpy array if needed
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector)
        
        # Ensure it's 2D for sklearn
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Handle NaN and Inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply scaling
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Select features used during training
        feature_vector_selected = feature_vector_scaled[:, self.feature_indices]
        
        # Apply PCA if it was used during training
        if self.pca is not None:
            feature_vector_final = self.pca.transform(feature_vector_selected)
        else:
            feature_vector_final = feature_vector_selected
        
        return feature_vector_final
    
    def predict(self, feature_vector):
        """
        Make prediction on input feature vector
        
        Args:
            feature_vector: Input feature vector
            
        Returns:
            Dictionary containing prediction and probability
        """
        # Preprocess the feature vector
        X_processed = self.preprocess_features(feature_vector)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        prediction_proba = self.model.predict_proba(X_processed)[0][1]
        
        return {
            'prediction': int(prediction),
            'probability': float(prediction_proba)
        }
    
    def create_ccmc_json(self, input_dt, prediction_result, output_path=None, issueTime=None):
        """
        Create CCMC JSON format output
        
        Args:
            input_dt: Input datetime (datetime object or string)
            prediction_result: Dictionary from predict() method
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
        ccmc_json = {
            "sep_forecast_submission": {
                "model": {
                    "short_name": "MagPy_ML_SHARP_HMI_CEA_numeric",
                    "spase_id": "spase://CCMC/SimulationModel/MagPy-ML/numeric/v1"
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
    
    def predict_and_export(self, feature_vector, input_dt, output_dir='predictions/numeric', issueTime=None):
        """
        Make prediction and export CCMC JSON in one step
        
        Args:
            feature_vector: Input feature vector
            input_dt: Input datetime
            output_dir: Directory to save JSON files
            issueTime: Issue time (optional)
            
        Returns:
            Dictionary containing prediction result and file path
        """
        # Make prediction
        prediction_result = self.predict(feature_vector)
        
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
        
        # Create filename
        filename = f'MagPy-ML-HMI-SHARP-Vector-numeric.{dt_parsed.year:04d}{dt_parsed.month:02d}{dt_parsed.day:02d}T{dt_parsed.hour:02d}{dt_parsed.minute:02d}.{issueTime.year:04d}{issueTime.month:02d}{issueTime.day:02d}T{issueTime.hour:02d}{issueTime.minute:02d}.json'
        output_path = os.path.join(output_dir, filename)
        
        # Create CCMC JSON
        ccmc_json = self.create_ccmc_json(input_dt, prediction_result, output_path, issueTime)
        
        return {
            'prediction': prediction_result['prediction'],
            'probability': prediction_result['probability'],
            'json_file': output_path,
            'ccmc_json': ccmc_json
        }

def main():
    """
    Example usage of the NumericPredictor
    """
    try:
        # Initialize predictor
        predictor = NumericPredictor()
        
        # ============================================================================
        # USER INPUT SECTION - MODIFY THIS SECTION WITH YOUR ACTUAL DATA
        # ============================================================================
        
        # Numeric feature vector components (based on NumericDataLoader.py):
        # 
        # One-time info (11 features):
        # ['Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 
        #  'Max Product of Half Angle and Speed of Recent CMEs', 'Number of Sunspots', 
        #  'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 
        #  'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares', 
        #  'Number of Recent SEPs', 'Number of Recent Subthreshold SEPs']
        
        # Per-blob vector columns (27 features per blob per timestep):
        # ['Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 
        #  'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 
        #  'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 
        #  'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 
        #  'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 
        #  'Volume Total Unsigned Magnetic Flux', 'Gradient_00', 'Gradient_10', 'Gradient_30', 
        #  'Gradient_50', 'Shear_00', 'Shear_10', 'Shear_30', 'Shear_50', 'Phi', 
        #  'Total Unsigned Current Helicity', 'Total Photospheric Magnetic Free Energy Density', 
        #  'Total Unsigned Vertical Current', 'Abs of Net Current helicity', 'Is Plage', 
        #  'Stonyhurst Longitude']
        
        # For per-disk-4hr granularity: 11 + (5 blobs × 6 timesteps × 27 features) = 11 + 810 = 821 features
        
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
            
            # Blob data for 5 blobs × 6 timesteps × 27 features (810 values total)
            # REPLACE ALL THESE ZEROS WITH YOUR ACTUAL DATA
            *([0.0] * 810)  # 810 zeros as placeholder - replace with your blob timeseries data
        ]
        
        # Your input datetime - REPLACE WITH YOUR ACTUAL DATETIME
        your_datetime = datetime.datetime(2023, 6, 15, 12, 0, 0)  # MODIFY THIS
        
        # ============================================================================
        # END USER INPUT SECTION
        # ============================================================================
        
        print(f"Feature vector length: {len(your_feature_vector)}")
        print("Expected length for numeric per-disk-4hr: 821")
        
        if len(your_feature_vector) != 821:
            print(f"WARNING: Feature vector length ({len(your_feature_vector)}) doesn't match expected length (821)")
            print("Please check your feature vector dimensions")
        
        # Make prediction and export
        result = predictor.predict_and_export(
            feature_vector=your_feature_vector,
            input_dt=your_datetime,
            output_dir='predictions/numeric'
        )
        
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"JSON file saved: {result['json_file']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()