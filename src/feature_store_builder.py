import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def create_feature_store_data():
    """
    Generates a synthetic dataset.
    Returns:
        pd.DataFrame: A DataFrame containing features and the target variable.
    """

    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_group1_features = 2
    n_group2_features = 10
    
    n_random_features = n_features - (n_group1_features + n_group2_features)
    group1_features = np.random.randn(n_samples, n_group1_features)
    group2_features = np.dot(group1_features, np.random.rand(n_group1_features, n_group2_features))
    group3_features = np.random.randn(n_samples, n_random_features)
    X = np.hstack([group1_features, group2_features, group3_features])
    y = (group1_features[:, 0] + group1_features[:, 1] > 0).astype(int)

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['target'] = y

    return features_df

def save_to_feature_store(df, db_path='sqlite:///feature_store/feature_store.db', table_name='features'):
    """
    Saves the DataFrame to a SQL database (Offline Store).
    
    Args:
        df (pd.DataFrame): Data to save.
        db_path (str): SQLAlchemy connection string.
        table_name (str): Name of the table in the database.
    """
    
    # Create the database engine
    engine = create_engine(db_path)
    
    # Save data to SQL
    # 'replace' drops the table if exists. In production, we might use 'append' with versioning.
    df.to_sql(table_name, con=engine, index=False, if_exists='replace')
    
    print(f"Data successfully saved to table '{table_name}' in {db_path}")
    return db_path