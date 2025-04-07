import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pickle
import shap
import os

def load_model_and_scaler(model_path, scaler_path):
    """
    Load the trained model and scaler from files
    
    I have done it because loading saved models avoids retraining
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_transaction(model, scaler, transaction_data):
    """
    Make a prediction for a single transaction
    
    I have done it because we need to process new transactions in the same way as training data
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(transaction_data, dict):
        transaction_data = pd.DataFrame([transaction_data])
    
    # Scale the data
    scaled_data = scaler.transform(transaction_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    return prediction, probability

def explain_prediction(model, scaler, transaction_data, feature_names):
    """
    Explain a prediction using SHAP values
    
    I have done it because model explainability is crucial for trust and debugging
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(transaction_data, dict):
        transaction_data = pd.DataFrame([transaction_data])
    
    # Scale the data
    scaled_data = scaler.transform(transaction_data)
    
    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, scaled_data)
    shap_values = explainer.shap_values(scaled_data)
    
    # Get feature contributions
    feature_contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': transaction_data.values[0],
        'Contribution': shap_values[0],
        'AbsContribution': np.abs(shap_values[0])
    })
    
    # Sort by absolute contribution
    feature_contributions = feature_contributions.sort_values('AbsContribution', ascending=False)
    
    return feature_contributions, explainer.expected_value, shap_values[0]

def plot_roc_curve(y_true, y_prob, ax=None):
    """
    Plot ROC curve
    
    I have done it because ROC curves help visualize model performance at different thresholds
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    return ax, roc_auc

def plot_precision_recall_curve(y_true, y_prob, ax=None):
    """
    Plot precision-recall curve
    
    I have done it because precision-recall curves are important for imbalanced datasets
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Plot
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    
    return ax, pr_auc

def plot_feature_importance(model, feature_names, top_n=10, ax=None):
    """
    Plot feature importance
    
    I have done it because understanding feature importance helps interpret the model
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get feature importance
    importance = np.abs(model.coef_[0])
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort and get top N
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title(f'Top {top_n} Feature Importance')
    
    return ax

def analyze_data_distribution(data, feature, by_class=True, ax=None):
    """
    Analyze and plot the distribution of a feature
    
    I have done it because understanding data distributions helps in feature engineering
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if by_class and 'Class' in data.columns:
        # Plot distribution by class
        sns.histplot(data=data, x=feature, hue='Class', kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature} by Class')
    else:
        # Plot overall distribution
        sns.histplot(data=data, x=feature, kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
    
    return ax

