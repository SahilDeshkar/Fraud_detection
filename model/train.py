import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

def load_data(file_path=None):
    """
    Load data from a file or generate synthetic data if no file is provided
    
    I have done it because we need a fallback option when no real data is available
    """
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Generate synthetic data
        print("No data file found. Generating synthetic data...")
        data = pd.DataFrame({
            'Time': np.random.randint(0, 172800, 1000),
            'Amount': np.random.exponential(scale=100, size=1000),
            'V1': np.random.normal(0, 1, 1000),
            'V2': np.random.normal(0, 1, 1000),
            'V3': np.random.normal(0, 1, 1000),
            'V4': np.random.normal(0, 1, 1000),
            'V5': np.random.normal(0, 1, 1000),
        })
        
        # Generate target variable (fraud or not)
        fraud_indices = np.random.choice(range(1000), size=50, replace=False)
        data['Class'] = 0
        data.loc[fraud_indices, 'Class'] = 1
        
        # Make fraudulent transactions have different feature distributions
        data.loc[data['Class'] == 1, 'V1'] = np.random.normal(-3, 1, len(data[data['Class'] == 1]))
        data.loc[data['Class'] == 1, 'V3'] = np.random.normal(3, 1, len(data[data['Class'] == 1]))
        
        return data

def preprocess_data(df):
    """
    Preprocess the data for model training
    
    I have done it because raw data needs to be transformed before model training
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model_with_grid_search(X_train, y_train):
    """
    Train a logistic regression model with grid search for hyperparameter tuning
    
    I have done it because optimizing hyperparameters improves model performance
    """
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced'],
        'solver': ['liblinear', 'saga']
    }
    
    # Create logistic regression model
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    I have done it because model evaluation is crucial to assess performance
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Print results
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    return cm, report, roc_auc

def save_model(model, scaler, output_dir='../model'):
    """
    Save the trained model and scaler
    
    I have done it because saving models allows for reuse without retraining
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    with open(os.path.join(output_dir, 'fraud_detection_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model and scaler saved to {output_dir}")

def main(data_path=None):
    """
    Main function to train and evaluate the model
    
    I have done it because organizing the workflow in a main function improves readability
    """
    # Load data
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Class distribution:\n{data['Class'].value_counts()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    print("Data preprocessing completed")
    
    # Train model
    print("Training model with grid search...")
    model = train_model_with_grid_search(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, scaler)

if __name__ == "__main__":
    main()

