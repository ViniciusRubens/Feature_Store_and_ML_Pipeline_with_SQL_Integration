import os
import json
import time
import feature_store_builder # Módulo atualizado
import model_training
import artifact_manager
import data_exploration
import pandas as pd
from sqlalchemy import create_engine

def main():
    print("\nStarting Pipeline Execution!")
    time.sleep(2)

    os.makedirs('feature_store', exist_ok=True)
    os.makedirs('pipeline_runs', exist_ok=True)

    # --- Step 1: Create & Persist Feature Store (SQL) ---
    print("Step 1: Creating and Persisting Feature Store in SQL...")
    
    features_df = feature_store_builder.create_feature_store_data()
    
    db_connection_str = 'sqlite:///feature_store/feature_store.db'
    feature_store_builder.save_to_feature_store(features_df, db_path=db_connection_str)

    print(">> Verifying data integrity by reading back from SQL...")
    engine = create_engine(db_connection_str)
    loaded_df = pd.read_sql("SELECT * FROM features", con=engine)

    # --- Step 2: Explore Data ---
    print("Step 2: Exploring Data and Generating Artifacts...")
    data_exploration.analyze_data(loaded_df, artifacts_path='pipeline_runs')

    # --- Step 3: Train and Evaluate Model ---
    print("Step 3: Training Model...")
    model, X_test, y_test, predictions, accuracy, class_report = model_training.train_evaluate_model(loaded_df)

    # Define paths for artifacts
    model_path = 'pipeline_runs/random_forest_model.joblib'
    predictions_path = 'pipeline_runs/predictions.csv'
    pipeline_run_info_path = 'pipeline_runs/pipeline_run_info.json'

    # --- Step 4: Save Artifacts ---
    print("Step 4: Saving Artifacts...")
    artifact_manager.save_model(model, model_path)
    artifact_manager.save_predictions(predictions, y_test, predictions_path)

    pipeline_run_info = {
        'model_type': 'RandomForestClassifier',
        'model_path': model_path,
        'feature_store_uri': db_connection_str,
        'source_table': 'features',
        'model_accuracy': accuracy
    }
    
    artifact_manager.save_run_info(pipeline_run_info, pipeline_run_info_path)

    # Final Output
    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(class_report)
    print("\nProject Completed Successfully.\n")
    
if __name__ == "__main__":
    main()