# SEP Final Models and Prediction Scripts

This collection includes scripts for training final SEP models without holdouts and making predictions with CCMC JSON output.

## Files Overview

### Training Script
- **`TrainFinalModels.py`** - Trains final models on all available data without holdouts

### Prediction Scripts
- **`PhotosphericPredictor.py`** - Makes predictions using the photospheric model
- **`CoronalPredictor.py`** - Makes predictions using the coronal model  
- **`NumericPredictor.py`** - Makes predictions using the numeric model

### Example Script
- **`example_usage.py`** - Comprehensive examples of how to use all prediction scripts

## Quick Start

### 1. Train Final Models

First, train the final models on all available data:

```bash
python TrainFinalModels.py
```

This will:
- Load all data from `../OutputData/UnifiedActiveRegionData_with_updated_SEP_list_but_no_line_count.csv`
- Train optimized models for photospheric, coronal, and numeric data subsets
- Save trained models to `final_models/` directory:
  - `final_models/photospheric_final_model.joblib`
  - `final_models/coronal_final_model.joblib`
  - `final_models/numeric_final_model.joblib`

### 2. Make Predictions

#### Individual Predictions

```python
from PhotosphericPredictor import PhotosphericPredictor
import numpy as np
import datetime

# Initialize predictor
predictor = PhotosphericPredictor()

# Your feature vector (replace with actual data)
features = np.array([...])  # Your feature vector

# Input datetime
input_time = datetime.datetime(2023, 6, 15, 12, 0, 0)

# Make prediction and export JSON
result = predictor.predict_and_export(
    feature_vector=features,
    input_dt=input_time,
    output_dir='predictions/photospheric'
)

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.4f}")
print(f"JSON saved: {result['json_file']}")
```

#### Batch Processing

```python
# Run example script to see all usage patterns
python example_usage.py
```

## Feature Vector Specifications

Each model expects specific feature vector dimensions and formats:

### Photospheric Model (521 features total)
- **One-time info (11 features):**
  - Number of Recent Flares
  - Max Class Type of Recent Flares
  - Number of Recent CMEs
  - Max Product of Half Angle and Speed of Recent CMEs
  - Number of Sunspots
  - Max Flare Peak of Recent Flares
  - Min Temperature of Recent Flares
  - Median Emission Measure of Recent Flares
  - Median Duration of Recent Flares
  - Number of Recent SEPs
  - Number of Recent Subthreshold SEPs

- **Blob timeseries data (510 features):**
  - 5 blobs × 6 timesteps × 17 features per blob per timestep
  - Features per blob: Latitude, Carrington Longitude, Gradient_00, Gradient_10, Gradient_30, Gradient_50, Shear_00, Shear_10, Shear_30, Shear_50, Phi, Total Unsigned Current Helicity, Total Photospheric Magnetic Free Energy Density, Total Unsigned Vertical Current, Abs of Net Current helicity, Is Plage, Stonyhurst Longitude

### Coronal Model (431 features total)
- **One-time info (11 features):** Same as photospheric
- **Blob timeseries data (420 features):**
  - 5 blobs × 6 timesteps × 14 features per blob per timestep
  - Features per blob: Latitude, Carrington Longitude, Volume Total Magnetic Energy, Volume Total Unsigned Current Helicity, Volume Total Absolute Net Current Helicity, Volume Mean Shear Angle, Volume Total Unsigned Volume Vertical Current, Volume Twist Parameter Alpha, Volume Mean Gradient of Vertical Magnetic Field, Volume Mean Gradient of Total Magnetic Field, Volume Total Magnitude of Lorentz Force, Volume Total Unsigned Magnetic Flux, Is Plage, Stonyhurst Longitude

### Numeric Model (821 features total)
- **One-time info (11 features):** Same as above
- **Blob timeseries data (810 features):**
  - 5 blobs × 6 timesteps × 27 features per blob per timestep
  - Features per blob: Latitude, Carrington Longitude, Volume Total Magnetic Energy, Volume Total Unsigned Current Helicity, Volume Total Absolute Net Current Helicity, Volume Mean Shear Angle, Volume Total Unsigned Volume Vertical Current, Volume Twist Parameter Alpha, Volume Mean Gradient of Vertical Magnetic Field, Volume Mean Gradient of Total Magnetic Field, Volume Total Magnitude of Lorentz Force, Volume Total Unsigned Magnetic Flux, Gradient_00, Gradient_10, Gradient_30, Gradient_50, Shear_00, Shear_10, Shear_30, Shear_50, Phi, Total Unsigned Current Helicity, Total Photospheric Magnetic Free Energy Density, Total Unsigned Vertical Current, Abs of Net Current helicity, Is Plage, Stonyhurst Longitude

## Model Details

### Optimal Hyperparameters Used

The final models use the following optimized hyperparameters from validation:

**Photospheric Model:**
- Model Type: Random Forest Complex
- Granularity: per-disk-4hr
- Oversampling Ratio: 0.55
- Feature Count: 90
- PCA: None

**Coronal Model:**
- Model Type: Random Forest Complex
- Granularity: per-disk-4hr  
- Oversampling Ratio: 0.7
- Feature Count: 70
- PCA: None

**Numeric Model:**
- Model Type: Random Forest Complex
- Granularity: per-disk-4hr
- Oversampling Ratio: 0.5
- Feature Count: 60
- PCA: None

## Prediction Classes

Each predictor class provides the following methods:

### Core Methods
- **`load_model()`** - Load trained model and preprocessing artifacts
- **`preprocess_features(feature_vector)`** - Preprocess input features
- **`predict(feature_vector)`** - Make prediction, returns dict with prediction and probability
- **`create_ccmc_json(input_dt, prediction_result, output_path, issueTime)`** - Create CCMC JSON format
- **`predict_and_export(feature_vector, input_dt, output_dir, issueTime)`** - One-step prediction and export

### Input Formats

**Feature Vectors:**
- Numpy arrays or Python lists
- Must match the exact dimensionality expected by each model
- NaN/Inf values are automatically handled

**Datetime Inputs:**
- `datetime.datetime` objects
- String formats: `"20230615_120000_TAI"` or `"2023-06-15 12:00:00"`

## Modifying Feature Vectors

To use the prediction scripts with your actual data:

1. **Open the predictor script** (e.g., `PhotosphericPredictor.py`)
2. **Find the main() function** at the bottom of the file
3. **Replace the placeholder values** in `your_feature_vector` with your actual data
4. **Update the datetime** in `your_datetime` 
5. **Run the script**: `python PhotosphericPredictor.py`

Example for photospheric model:
```python
your_feature_vector = [
    # One-time info (11 values) - REPLACE WITH YOUR DATA
    2.0,    # Number of Recent Flares (your actual value)
    5.5,    # Max Class Type of Recent Flares (your actual value)
    1.0,    # Number of Recent CMEs (your actual value)
    # ... continue with your actual values
    
    # Blob timeseries data (510 values) - REPLACE WITH YOUR DATA
    *your_actual_blob_data  # Your 510 blob timeseries values
]
```

## CCMC JSON Output Format

The prediction scripts generate JSON files in CCMC-compliant format:

```json
{
  "sep_forecast_submission": {
    "model": {
      "short_name": "MagPy_ML_SHARP_HMI_CEA_photospheric",
      "spase_id": "spase://CCMC/SimulationModel/MagPy-ML/photospheric/v1"
    },
    "mode": "forecast",
    "issue_time": "2023-06-15T12:00:00Z",
    "inputs": [...],
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
          "start_time": "2023-06-15T12:00:00Z",
          "end_time": "2023-06-16T12:00:00Z"
        },
        "probabilities": [
          {
            "probability_value": "0.15432",
            "threshold": 10,
            "threshold_units": "pfu"
          }
        ],
        "all_clear": {
          "all_clear_boolean": true,
          "threshold": 10,
          "threshold_units": "pfu",
          "probability_threshold": 0.5
        }
      }
    ]
  }
}
```

## Directory Structure

After running the scripts, you'll have:

```
├── final_models/
│   ├── photospheric_final_model.joblib
│   ├── coronal_final_model.joblib
│   └── numeric_final_model.joblib
├── predictions/
│   ├── photospheric/
│   ├── coronal/
│   └── numeric/
└── [script files]
```

## Usage Examples

The `example_usage.py` script demonstrates:

1. **Single Prediction** - Basic usage with one feature vector
2. **Batch Predictions** - Processing multiple feature vectors
3. **Different Datetime Formats** - Various input datetime formats
4. **Prediction Only** - Getting predictions without saving JSON
5. **Ensemble Prediction** - Using multiple models together

⚠️ **Note:** Example scripts use placeholder feature vectors filled with zeros. You must replace these with your actual data.

## Error Handling

All scripts include comprehensive error handling:
- Missing model files
- Invalid feature vector dimensions
- Malformed datetime inputs
- File I/O errors

## Dependencies

Required Python packages:
- `numpy`
- `pandas` 
- `scikit-learn`
- `joblib`
- `imbalanced-learn`

## Notes

- Models are trained using all available data (no holdouts) for maximum performance
- Feature preprocessing (scaling, selection, PCA) is applied automatically
- JSON files are named with timestamp information for easy tracking
- All three models use the same general architecture but with data-subset-specific optimization

## Troubleshooting

**Model not found errors:**
- Ensure you've run `TrainFinalModels.py` first
- Check that model files exist in `final_models/` directory

**Feature dimension errors:**
- Verify your feature vector has the correct number of elements
- Check that features are in the same order as training data
- See feature specifications above for exact requirements

**Datetime parsing errors:**
- Use supported formats: datetime objects, TAI strings, or ISO strings
- Ensure datetime values are valid
