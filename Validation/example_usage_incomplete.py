"""
Example Usage Script for SEP Predictors

This script demonstrates how to use the three SEP prediction models:
- PhotosphericPredictor
- CoronalPredictor  
- NumericPredictor

Each predictor loads its respective trained model and makes predictions on feature vectors,
outputting results in CCMC JSON format.
"""

import numpy as np
import datetime
import os
import sys

# Add current directory to path to import predictors
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PhotosphericPredictor import PhotosphericPredictor
from CoronalPredictor import CoronalPredictor
from NumericPredictor import NumericPredictor

def example_single_prediction():
    """
    Example of making a single prediction with each model
    """
    print("="*60)
    print("EXAMPLE: Single Prediction with Each Model")
    print("="*60)
    
    # Example datetime
    input_datetime = datetime.datetime(2023, 6, 15, 12, 0, 0)
    
    print("\n⚠️  NOTE: This example uses placeholder feature vectors filled with zeros.")
    print("    You MUST replace these with your actual feature vectors for real predictions.")
    print("    See individual predictor scripts for detailed feature vector specifications.")
    
    # Feature vector specifications based on data loaders:
    # Photospheric: 521 features (11 one-time + 5 blobs × 6 timesteps × 17 features)
    # Coronal: 431 features (11 one-time + 5 blobs × 6 timesteps × 14 features)  
    # Numeric: 821 features (11 one-time + 5 blobs × 6 timesteps × 27 features)
    
    photospheric_features = np.zeros(521)    # REPLACE with actual photospheric features
    coronal_features = np.zeros(431)         # REPLACE with actual coronal features
    numeric_features = np.zeros(821)         # REPLACE with actual numeric features
    
    try:
        # Photospheric prediction
        print("\n--- Photospheric Model ---")
        print("Expected feature vector length: 521")
        photo_predictor = PhotosphericPredictor()
        photo_result = photo_predictor.predict_and_export(
            feature_vector=photospheric_features,
            input_dt=input_datetime,
            output_dir='predictions/photospheric'
        )
        print(f"Photospheric - Prediction: {photo_result['prediction']}, Probability: {photo_result['probability']:.4f}")
        print(f"JSON saved to: {photo_result['json_file']}")
        
        # Coronal prediction
        print("\n--- Coronal Model ---")
        print("Expected feature vector length: 431")
        coronal_predictor = CoronalPredictor()
        coronal_result = coronal_predictor.predict_and_export(
            feature_vector=coronal_features,
            input_dt=input_datetime,
            output_dir='predictions/coronal'
        )
        print(f"Coronal - Prediction: {coronal_result['prediction']}, Probability: {coronal_result['probability']:.4f}")
        print(f"JSON saved to: {coronal_result['json_file']}")
        
        # Numeric prediction
        print("\n--- Numeric Model ---")
        print("Expected feature vector length: 821")
        numeric_predictor = NumericPredictor()
        numeric_result = numeric_predictor.predict_and_export(
            feature_vector=numeric_features,
            input_dt=input_datetime,
            output_dir='predictions/numeric'
        )
        print(f"Numeric - Prediction: {numeric_result['prediction']}, Probability: {numeric_result['probability']:.4f}")
        print(f"JSON saved to: {numeric_result['json_file']}")
        
    except Exception as e:
        print(f"Error in single prediction example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_batch_predictions():
    """
    Example of making predictions on multiple feature vectors
    """
    print("\n" + "="*60)
    print("EXAMPLE: Batch Predictions")
    print("="*60)
    
    # Example datetimes
    datetimes = [
        datetime.datetime(2023, 6, 15, 12, 0, 0),
        datetime.datetime(2023, 6, 16, 6, 0, 0),
        datetime.datetime(2023, 6, 17, 18, 0, 0)
    ]
    
    print("\n⚠️  NOTE: This example uses placeholder feature vectors filled with zeros.")
    print("    You MUST replace these with your actual feature vectors for real predictions.")
    
    try:
        # Initialize predictors
        photo_predictor = PhotosphericPredictor()
        coronal_predictor = CoronalPredictor()
        numeric_predictor = NumericPredictor()
        
        for i, dt in enumerate(datetimes):
            print(f"\n--- Prediction {i+1} for {dt} ---")
            
            # Generate placeholder feature vectors (REPLACE with actual data)
            photo_features = np.zeros(521)    # REPLACE with actual photospheric features
            coronal_features = np.zeros(431)  # REPLACE with actual coronal features
            numeric_features = np.zeros(821)  # REPLACE with actual numeric features
            
            # Make predictions
            photo_pred = photo_predictor.predict(photo_features)
            coronal_pred = coronal_predictor.predict(coronal_features)
            numeric_pred = numeric_predictor.predict(numeric_features)
            
            print(f"Photospheric: {photo_pred['prediction']} (prob: {photo_pred['probability']:.4f})")
            print(f"Coronal:      {coronal_pred['prediction']} (prob: {coronal_pred['probability']:.4f})")
            print(f"Numeric:      {numeric_pred['prediction']} (prob: {numeric_pred['probability']:.4f})")
            
    except Exception as e:
        print(f"Error in batch prediction example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_custom_datetime_formats():
    """
    Example of using different datetime input formats
    """
    print("\n" + "="*60)
    print("EXAMPLE: Different Datetime Input Formats")
    print("="*60)
    
    try:
        predictor = PhotosphericPredictor()
        features = np.zeros(521)  # REPLACE with actual photospheric features
        
        # Different datetime formats
        datetime_formats = [
            datetime.datetime(2023, 6, 15, 12, 0, 0),  # datetime object
            "20230615_120000_TAI",                      # TAI format string
            "2023-06-15 12:00:00",                      # Standard format string
        ]
        
        print("\n⚠️  NOTE: Using placeholder feature vector. Replace with actual data.")
        
        for i, dt_input in enumerate(datetime_formats):
            print(f"\nFormat {i+1}: {dt_input} (type: {type(dt_input).__name__})")
            
            result = predictor.predict_and_export(
                feature_vector=features,
                input_dt=dt_input,
                output_dir=f'predictions/datetime_test_{i+1}'
            )
            
            print(f"Prediction: {result['prediction']}, Probability: {result['probability']:.4f}")
            print(f"JSON file: {result['json_file']}")
            
    except Exception as e:
        print(f"Error in datetime format example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_just_prediction_no_export():
    """
    Example of making predictions without exporting JSON files
    """
    print("\n" + "="*60)
    print("EXAMPLE: Predictions Only (No JSON Export)")
    print("="*60)
    
    try:
        # Initialize all predictors
        predictors = {
            'photospheric': PhotosphericPredictor(),
            'coronal': CoronalPredictor(),
            'numeric': NumericPredictor()
        }
        
        # Example feature vectors (REPLACE with actual data)
        features = {
            'photospheric': np.zeros(521),   # REPLACE with actual photospheric features
            'coronal': np.zeros(431),        # REPLACE with actual coronal features
            'numeric': np.zeros(821),        # REPLACE with actual numeric features
        }
        
        print("\n⚠️  NOTE: Using placeholder feature vectors. Replace with actual data.")
        print("\nMaking predictions without saving JSON files:")
        for model_type, predictor in predictors.items():
            result = predictor.predict(features[model_type])
            print(f"{model_type.capitalize():12}: Prediction={result['prediction']}, Probability={result['probability']:.4f}")
            
    except Exception as e:
        print(f"Error in prediction-only example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_ensemble_prediction():
    """
    Example of using all three models together for ensemble prediction
    """
    print("\n" + "="*60)
    print("EXAMPLE: Ensemble Prediction (All Three Models)")
    print("="*60)
    
    try:
        # Initialize all predictors
        photo_predictor = PhotosphericPredictor()
        coronal_predictor = CoronalPredictor()
        numeric_predictor = NumericPredictor()
        
        # Example input
        input_datetime = datetime.datetime(2023, 6, 15, 12, 0, 0)
        
        # Example feature vectors (REPLACE with actual data)
        photo_features = np.zeros(521)    # REPLACE with actual photospheric features
        coronal_features = np.zeros(431)  # REPLACE with actual coronal features
        numeric_features = np.zeros(821)  # REPLACE with actual numeric features
        
        print("\n⚠️  NOTE: Using placeholder feature vectors. Replace with actual data.")
        
        # Make predictions with all models
        photo_result = photo_predictor.predict(photo_features)
        coronal_result = coronal_predictor.predict(coronal_features)
        numeric_result = numeric_predictor.predict(numeric_features)
        
        # Combine results
        probabilities = [
            photo_result['probability'],
            coronal_result['probability'],
            numeric_result['probability']
        ]
        
        predictions = [
            photo_result['prediction'],
            coronal_result['prediction'],
            numeric_result['prediction']
        ]
        
        # Simple ensemble methods
        avg_probability = np.mean(probabilities)
        majority_vote = int(np.sum(predictions) >= 2)  # Majority vote
        max_probability = np.max(probabilities)
        
        print(f"\nIndividual Model Results:")
        print(f"Photospheric: Prediction={photo_result['prediction']}, Probability={photo_result['probability']:.4f}")
        print(f"Coronal:      Prediction={coronal_result['prediction']}, Probability={coronal_result['probability']:.4f}")
        print(f"Numeric:      Prediction={numeric_result['prediction']}, Probability={numeric_result['probability']:.4f}")
        
        print(f"\nEnsemble Results:")
        print(f"Average Probability:    {avg_probability:.4f}")
        print(f"Maximum Probability:    {max_probability:.4f}")
        print(f"Majority Vote:          {majority_vote}")
        print(f"Unanimous Agreement:    {len(set(predictions)) == 1}")
        
        # Export ensemble result as JSON for the model with highest probability
        best_model_idx = np.argmax(probabilities)
        model_names = ['photospheric', 'coronal', 'numeric']
        predictors = [photo_predictor, coronal_predictor, numeric_predictor]
        features_list = [photo_features, coronal_features, numeric_features]
        
        best_model = model_names[best_model_idx]
        best_predictor = predictors[best_model_idx]
        
        print(f"\nExporting JSON for best performing model: {best_model}")
        ensemble_result = best_predictor.predict_and_export(
            feature_vector=features_list[best_model_idx],
            input_dt=input_datetime,
            output_dir=f'predictions/ensemble_{best_model}'
        )
        print(f"Ensemble JSON saved to: {ensemble_result['json_file']}")
        
    except Exception as e:
        print(f"Error in ensemble prediction example: {str(e)}")
        import traceback
        traceback.print_exc()

def check_model_availability():
    """
    Check which trained models are available
    """
    print("\n" + "="*60)
    print("CHECKING MODEL AVAILABILITY")
    print("="*60)
    
    model_paths = {
        'photospheric': 'final_models/photospheric_final_model.joblib',
        'coronal': 'final_models/coronal_final_model.joblib',
        'numeric': 'final_models/numeric_final_model.joblib'
    }
    
    available_models = []
    
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            print(f"✓ {model_type.capitalize()} model: AVAILABLE ({path})")
            available_models.append(model_type)
        else:
            print(f"✗ {model_type.capitalize()} model: NOT FOUND ({path})")
    
    if not available_models:
        print("\n⚠️  No trained models found!")
        print("Please run TrainFinalModels.py first to train the models.")
        return False
    else:
        print(f"\n✓ {len(available_models)} model(s) available: {', '.join(available_models)}")
        return True

def main():
    """
    Main function demonstrating all examples
    """
    print("SEP PREDICTION MODELS - USAGE EXAMPLES")
    print("="*60)
    
    # Check if models are available
    if not check_model_availability():
        print("\nCannot run examples without trained models.")
        print("Please ensure you have run TrainFinalModels.py first.")
        return
    
    print("\nRunning usage examples...")
    
    # Run all examples
    try:
        example_single_prediction()
        example_batch_predictions()
        example_custom_datetime_formats()
        example_just_prediction_no_export()
        example_ensemble_prediction()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated prediction files can be found in the 'predictions/' directory.")
        print("\nFeature Vector Specifications:")
        print("- Photospheric: 521 features")
        print("- Coronal:      431 features")
        print("- Numeric:      821 features")
        print("\n⚠️  Remember to replace placeholder feature vectors with your actual data!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
        
    except Exception as e:
        print(f"Error in ensemble prediction example: {str(e)}")
        import traceback
        traceback.print_exc()000)       # Replace with actual numeric features
    
    try:
        # Photospheric prediction
        print("\n--- Photospheric Model ---")
        photo_predictor = PhotosphericPredictor()
        photo_result = photo_predictor.predict_and_export(
            feature_vector=photospheric_features,
            input_dt=input_datetime,
            output_dir='predictions/photospheric'
        )
        print(f"Photospheric - Prediction: {photo_result['prediction']}, Probability: {photo_result['probability']:.4f}")
        print(f"JSON saved to: {photo_result['json_file']}")
        
        # Coronal prediction
        print("\n--- Coronal Model ---")
        coronal_predictor = CoronalPredictor()
        coronal_result = coronal_predictor.predict_and_export(
            feature_vector=coronal_features,
            input_dt=input_datetime,
            output_dir='predictions/coronal'
        )
        print(f"Coronal - Prediction: {coronal_result['prediction']}, Probability: {coronal_result['probability']:.4f}")
        print(f"JSON saved to: {coronal_result['json_file']}")
        
        # Numeric prediction
        print("\n--- Numeric Model ---")
        numeric_predictor = NumericPredictor()
        numeric_result = numeric_predictor.predict_and_export(
            feature_vector=numeric_features,
            input_dt=input_datetime,
            output_dir='predictions/numeric'
        )
        print(f"Numeric - Prediction: {numeric_result['prediction']}, Probability: {numeric_result['probability']:.4f}")
        print(f"JSON saved to: {numeric_result['json_file']}")
        
    except Exception as e:
        print(f"Error in prediction-only example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_ensemble_prediction():
    """
    Example of using all three models together for ensemble prediction
    """
    print("\n" + "="*60)
    print("EXAMPLE: Ensemble Prediction (All Three Models)")
    print("="*60)
    
    try:
        # Initialize all predictors
        photo_predictor = PhotosphericPredictor()
        coronal_predictor = CoronalPredictor()
        numeric_predictor = NumericPredictor()
        
        # Example input
        input_datetime = datetime.datetime(2023, 6, 15, 12, 0, 0)
        
        # Example feature vectors (replace with actual data)
        photo_features = np.random.randn(1000)
        coronal_features = np.random.randn(1000)
        numeric_features = np.random.randn(1000)
        
        # Make predictions with all models
        photo_result = photo_predictor.predict(photo_features)
        coronal_result = coronal_predictor.predict(coronal_features)
        numeric_result = numeric_predictor.predict(numeric_features)
        
        # Combine results
        probabilities = [
            photo_result['probability'],
            coronal_result['probability'], 
            numeric_result['probability']
        ]
        
        predictions = [
            photo_result['prediction'],
            coronal_result['prediction'],
            numeric_result['prediction']
        ]
        
        # Simple ensemble methods
        avg_probability = np.mean(probabilities)
        majority_vote = int(np.sum(predictions) >= 2)  # Majority vote
        max_probability = np.max(probabilities)
        
        print(f"\nIndividual Model Results:")
        print(f"Photospheric: Prediction={photo_result['prediction']}, Probability={photo_result['probability']:.4f}")
        print(f"Coronal:      Prediction={coronal_result['prediction']}, Probability={coronal_result['probability']:.4f}")
        print(f"Numeric:      Prediction={numeric_result['prediction']}, Probability={numeric_result['probability']:.4f}")
        
        print(f"\nEnsemble Results:")
        print(f"Average Probability:    {avg_probability:.4f}")
        print(f"Maximum Probability:    {max_probability:.4f}")
        print(f"Majority Vote:          {majority_vote}")
        print(f"Unanimous Agreement:    {len(set(predictions)) == 1}")
        
        # Export ensemble result as JSON for the model with highest probability
        best_model_idx = np.argmax(probabilities)
        model_names = ['photospheric', 'coronal', 'numeric']
        predictors = [photo_predictor, coronal_predictor, numeric_predictor]
        features_list = [photo_features, coronal_features, numeric_features]
        
        best_model = model_names[best_model_idx]
        best_predictor = predictors[best_model_idx]
        
        print(f"\nExporting JSON for best performing model: {best_model}")
        ensemble_result = best_predictor.predict_and_export(
            feature_vector=features_list[best_model_idx],
            input_dt=input_datetime,
            output_dir=f'predictions/ensemble_{best_model}'
        )
        print(f"Ensemble JSON saved to: {ensemble_result['json_file']}")
        
    except Exception as e:
        print(f"Error in ensemble prediction example: {str(e)}")
        import traceback
        traceback.print_exc()

def check_model_availability():
    """
    Check which trained models are available
    """
    print("\n" + "="*60)
    print("CHECKING MODEL AVAILABILITY")
    print("="*60)
    
    model_paths = {
        'photospheric': 'final_models/photospheric_final_model.joblib',
        'coronal': 'final_models/coronal_final_model.joblib',
        'numeric': 'final_models/numeric_final_model.joblib'
    }
    
    available_models = []
    
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            print(f"✓ {model_type.capitalize()} model: AVAILABLE ({path})")
            available_models.append(model_type)
        else:
            print(f"✗ {model_type.capitalize()} model: NOT FOUND ({path})")
    
    if not available_models:
        print("\n⚠️  No trained models found!")
        print("Please run TrainFinalModels.py first to train the models.")
        return False
    else:
        print(f"\n✓ {len(available_models)} model(s) available: {', '.join(available_models)}")
        return True

def main():
    """
    Main function demonstrating all examples
    """
    print("SEP PREDICTION MODELS - USAGE EXAMPLES")
    print("="*60)
    
    # Check if models are available
    if not check_model_availability():
        print("\nCannot run examples without trained models.")
        print("Please ensure you have run TrainFinalModels.py first.")
        return
    
    print("\nRunning usage examples...")
    
    # Run all examples
    try:
        example_single_prediction()
        example_batch_predictions()
        example_custom_datetime_formats()
        example_just_prediction_no_export()
        example_ensemble_prediction()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated prediction files can be found in the 'predictions/' directory.")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() single prediction example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_batch_predictions():
    """
    Example of making predictions on multiple feature vectors
    """
    print("\n" + "="*60)
    print("EXAMPLE: Batch Predictions")
    print("="*60)
    
    # Example datetimes
    datetimes = [
        datetime.datetime(2023, 6, 15, 12, 0, 0),
        datetime.datetime(2023, 6, 16, 6, 0, 0),
        datetime.datetime(2023, 6, 17, 18, 0, 0)
    ]
    
    try:
        # Initialize predictors
        photo_predictor = PhotosphericPredictor()
        coronal_predictor = CoronalPredictor()
        numeric_predictor = NumericPredictor()
        
        for i, dt in enumerate(datetimes):
            print(f"\n--- Prediction {i+1} for {dt} ---")
            
            # Generate example feature vectors (replace with actual data)
            photo_features = np.random.randn(1000)
            coronal_features = np.random.randn(1000)
            numeric_features = np.random.randn(1000)
            
            # Make predictions
            photo_pred = photo_predictor.predict(photo_features)
            coronal_pred = coronal_predictor.predict(coronal_features)
            numeric_pred = numeric_predictor.predict(numeric_features)
            
            print(f"Photospheric: {photo_pred['prediction']} (prob: {photo_pred['probability']:.4f})")
            print(f"Coronal:      {coronal_pred['prediction']} (prob: {coronal_pred['probability']:.4f})")
            print(f"Numeric:      {numeric_pred['prediction']} (prob: {numeric_pred['probability']:.4f})")
            
    except Exception as e:
        print(f"Error in batch prediction example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_custom_datetime_formats():
    """
    Example of using different datetime input formats
    """
    print("\n" + "="*60)
    print("EXAMPLE: Different Datetime Input Formats")
    print("="*60)
    
    try:
        predictor = PhotosphericPredictor()
        features = np.random.randn(1000)
        
        # Different datetime formats
        datetime_formats = [
            datetime.datetime(2023, 6, 15, 12, 0, 0),  # datetime object
            "20230615_120000_TAI",                      # TAI format string
            "2023-06-15 12:00:00",                      # Standard format string
        ]
        
        for i, dt_input in enumerate(datetime_formats):
            print(f"\nFormat {i+1}: {dt_input} (type: {type(dt_input).__name__})")
            
            result = predictor.predict_and_export(
                feature_vector=features,
                input_dt=dt_input,
                output_dir=f'predictions/datetime_test_{i+1}'
            )
            
            print(f"Prediction: {result['prediction']}, Probability: {result['probability']:.4f}")
            print(f"JSON file: {result['json_file']}")
            
    except Exception as e:
        print(f"Error in datetime format example: {str(e)}")
        import traceback
        traceback.print_exc()

def example_just_prediction_no_export():
    """
    Example of making predictions without exporting JSON files
    """
    print("\n" + "="*60)
    print("EXAMPLE: Predictions Only (No JSON Export)")
    print("="*60)
    
    try:
        # Initialize all predictors
        predictors = {
            'photospheric': PhotosphericPredictor(),
            'coronal': CoronalPredictor(),
            'numeric': NumericPredictor()
        }
        
        # Example feature vectors
        features = {
            'photospheric': np.random.randn(1000),
            'coronal': np.random.randn(1000),
            'numeric': np.random.randn(1000)
        }
        
        print("\nMaking predictions without saving JSON files:")
        for model_type, predictor in predictors.items():
            result = predictor.predict(features[model_type])
            print(f"{model_type.capitalize():12}: Prediction={result['prediction']}, Probability={result['probability']:.4f}")
            
    except Exception as e:
        print(f"Error in