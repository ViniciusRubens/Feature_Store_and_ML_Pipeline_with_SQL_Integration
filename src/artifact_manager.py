import json
import pandas as pd
from joblib import dump

def save_model(model, model_path):
    """Saves the trained model object using joblib."""

    dump(model, model_path)

def save_predictions(predictions, y_test, predictions_path):
    """Saves predictions vs actual values to a CSV file."""

    predictions_df = pd.DataFrame({
        'ActualValue': y_test, 
        'PredictedValue': predictions
    })
    predictions_df.to_csv(predictions_path, index=False)

def save_run_info(run_info, run_info_path):
    """Saves pipeline execution metadata to a JSON file."""
    
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f)