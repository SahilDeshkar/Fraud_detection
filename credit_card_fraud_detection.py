import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import shap
import io

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('/content/creditcard.csv')
    return data

data = load_data()

# Title and description
st.title('Credit Card Fraud Detection')
st.markdown("""
This application demonstrates a machine learning model for detecting credit card fraud using logistic regression.
The dataset contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.
""")

# Data overview
if st.checkbox('Show Data Overview'):
    st.write("**Dataset Information:**")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("**Class Distribution:**")
    st.write(data['Class'].value_counts())

# Dynamic Filtering
st.sidebar.subheader("Filter Transactions")
time_min, time_max = st.sidebar.slider("Select Time Range", int(data['Time'].min()), int(data['Time'].max()), (int(data['Time'].min()), int(data['Time'].max())))
amount_min, amount_max = st.sidebar.slider("Select Amount Range", float(data['Amount'].min()), float(data['Amount'].max()), (float(data['Amount'].min()), float(data['Amount'].max())))

filtered_data = data[(data['Time'] >= time_min) & (data['Time'] <= time_max) & 
                     (data['Amount'] >= amount_min) & (data['Amount'] <= amount_max)]

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
    cm = confusion_matrix(Y_test, X_test_prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    st.pyplot(fig)

plot_confusion_matrix(Y_test, X_test_prediction)

# ROC Curve
def plot_roc_curve():
    st.markdown("### ROC Curve")
    fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)

plot_roc_curve()

# Feature Importance with SHAP
def plot_shap_values():
    st.markdown("### Feature Importance with SHAP")
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot(fig)

plot_shap_values()

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
st.write(model_loaded.predict(X_train))
