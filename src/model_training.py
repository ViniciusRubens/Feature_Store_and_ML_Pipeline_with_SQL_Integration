from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_evaluate_model(features_df):
    """
    Splits data, trains a Random Forest model, and evaluates performance.
    
    Returns:
        tuple: model, X_test, y_test, predictions, accuracy, classification_report_str
    """
    
    # Split data into training and testing sets (80/20 split)
    X = features_df.iloc[:, :-1]
    y = features_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train RandomForest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    predictions = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    return rf_model, X_test, y_test, predictions, accuracy, class_report