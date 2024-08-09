import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.ensemble import IsolationForest
import shap
import pickle
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('/content/creditcard.csv')
    return data

data = load_data()

# Title and description
st.title('Credit Card Fraud Detection')
st.markdown("""
This application demonstrates a machine learning model for detecting credit card fraud using logistic regression and anomaly detection with Isolation Forest. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
""")

# Data overview
if st.checkbox('Show Data Overview'):
    st.write("**Dataset Information:**")
    st.write(data.describe())
    st.write("**Class Distribution:**")
    st.write(data['Class'].value_counts())

# Dynamic Filtering
st.sidebar.subheader("Filter Transactions")
time_min, time_max = st.sidebar.slider("Select Time Range", int(data['Time'].min()), int(data['Time'].max()), (int(data['Time'].min()), int(data['Time'].max())))
amount_min, amount_max = st.sidebar.slider("Select Amount Range", float(data['Amount'].min()), float(data['Amount'].max()), (float(data['Amount'].min()), float(data['Amount'].max())))

filtered_data = data[(data['Time'] >= time_min) & (data['Time'] <= time_max) & 
                     (data['Amount'] >= amount_min) & (data['Amount'] <= amount_max)]

# Data distribution
def plot_data_distribution(data):
    st.write("**Distribution of Transactions Before Sampling:**")
    sns.countplot(x='Class', data=data)
    st.pyplot()

    st.write("**Distribution of Transaction Amounts:**")
    sns.histplot(data=data[data['Class'] == 0], x='Amount', bins=50, color='blue', label='Legit', alpha=0.6)
    sns.histplot(data=data[data['Class'] == 1], x='Amount', bins=50, color='red', label='Fraud', alpha=0.6)
    st.pyplot()

plot_data_distribution(filtered_data)

# Sample the data
def sample_data(data, sample_size=492):
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(min(sample_size, legit.shape[0]), replace=False)
    fraud_sample = fraud.sample(min(sample_size, fraud.shape[0]), replace=False)
    return pd.concat([legit_sample, fraud_sample], axis=0)

new_data = sample_data(filtered_data)

# Prepare the feature and target variables
X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)

# Predictions and evaluations
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)
roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])

# Show metrics
st.markdown(f"### Model Performance")
st.write(f"- **Test Accuracy:** {test_accuracy:.2f}")
st.write(f"- **ROC AUC Score:** {roc_auc:.2f}")

# Confusion Matrix
def plot_confusion_matrix(Y_test, X_test_prediction):
    st.markdown("### Confusion Matrix")
    st.write("This matrix shows the number of correct and incorrect predictions made by the model.")
    cm = confusion_matrix(Y_test, X_test_prediction)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    st.pyplot()

plot_confusion_matrix(Y_test, X_test_prediction)

# ROC Curve
def plot_roc_curve():
    st.markdown("### ROC Curve")
    st.write("The ROC curve illustrates the true positive rate (recall) against the false positive rate, showing the trade-off between sensitivity and specificity.")
    fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
    sns.lineplot(x=fpr, y=tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    sns.lineplot(x=[0, 1], y=[0, 1], color='gray', linestyle='--')
    st.pyplot()

plot_roc_curve()

# Feature Importance with SHAP
def plot_shap_values():
    st.markdown("### Feature Importance with SHAP")
    st.write("This plot shows the impact of each feature on the prediction made by the model using SHAP values.")
    
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot()

plot_shap_values()

# Anomaly Detection with Isolation Forest
def plot_anomaly_detection():
    st.markdown("### Anomaly Detection with Isolation Forest")
    st.write("This plot shows the anomaly scores given by the Isolation Forest model for the transactions.")
    
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    data['Anomaly_Score'] = iso_forest.fit_predict(data.drop(columns=['Class']))
    
    sns.histplot(data=data[data['Anomaly_Score'] == -1], x='Amount', bins=50, color='red', label='Anomaly', alpha=0.6)
    sns.histplot(data=data[data['Anomaly_Score'] == 1], x='Amount', bins=50, color='blue', label='Normal', alpha=0.6)
    st.pyplot()

plot_anomaly_detection()

# Save and load model
def save_model(model):
    with open('/content/ML_project.pkl', 'wb') as file:
        pickle.dump(model, file)
    st.success("Model saved successfully!")

def load_model():
    with open('/content/ML_project.pkl', 'rb') as file:
        return pickle.load(file)

save_model(model)
model_loaded = load_model()

# Predictions with loaded model
st.markdown("### Predictions with Loaded Model")
st.write("Here are predictions made by the loaded model on the training data:")
st.write(model_loaded.predict(X_train))
