import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Set up the main title and description
st.title("Credit Card Fraud Detection System")
st.markdown("""
This application uses machine learning to detect potentially fraudulent credit card transactions.
Upload your transaction data or use our sample dataset to train a model and identify suspicious activities.
""")

# Function to load sample data
@st.cache_data
def load_sample_data():
    # I've created a simplified version of credit card transaction data
    # I have done this because real credit card datasets are often very large and imbalanced
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
    # I have done it this way to create an imbalanced dataset similar to real fraud data
    fraud_indices = np.random.choice(range(1000), size=50, replace=False)
    data['Class'] = 0
    data.loc[fraud_indices, 'Class'] = 1
    
    # Make fraudulent transactions have different feature distributions
    data.loc[data['Class'] == 1, 'V1'] = np.random.normal(-3, 1, len(data[data['Class'] == 1]))
    data.loc[data['Class'] == 1, 'V3'] = np.random.normal(3, 1, len(data[data['Class'] == 1]))
    
    return data

# Function to download the trained model
def download_model(model):
    # I have done it because users might want to save and reuse their trained model
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="fraud_detection_model.pkl">Download Trained Model</a>'
    return href

# Function to preprocess data
def preprocess_data(df):
    # I have done it because standardizing features improves logistic regression performance
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, X_train, X_test

# Function to train the model
def train_model(X_train, y_train, C=1.0, class_weight=None):
    # I have done it because logistic regression is effective for binary classification tasks
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # I have done it because comprehensive evaluation metrics help assess model performance
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return cm, fpr, tpr, roc_auc, precision, recall, report, y_prob

# Function to plot evaluation metrics
def plot_evaluation(cm, fpr, tpr, roc_auc, precision, recall, report):
    # I have done it because visualizations make it easier to understand model performance
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
    
    with col2:
        # ROC Curve
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, color='green', lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        st.pyplot(fig)

# Function to explain model predictions using coefficients and permutation importance
def explain_model(model, X_test, X_test_orig, feature_names):
    # Replaced SHAP with model coefficients and permutation importance
    st.subheader("Model Explainability")
    
    # Model coefficients
    st.write("Feature Importance Based on Model Coefficients")
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    coefficients['Absolute_Coefficient'] = abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Absolute_Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in coefficients['Coefficient']]
    sns.barplot(x='Absolute_Coefficient', y='Feature', data=coefficients, palette=colors, ax=ax)
    ax.set_title('Feature Importance Based on Coefficients')
    ax.set_xlabel('Absolute Coefficient Value')
    st.pyplot(fig)
    
    # Permutation importance
    st.write("Feature Importance Based on Permutation Importance")
    with st.spinner("Calculating permutation importance..."):
        # Calculate permutation importance on a subset for efficiency
        perm_importance = permutation_importance(model, X_test[:100], X_test_orig['Class'][:100], 
                                               n_repeats=5, random_state=42)
        
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=perm_df, color='blue', ax=ax)
    ax.set_title('Feature Importance Based on Permutation Importance')
    st.pyplot(fig)

# Main application flow
tab1, tab2, tab3 = st.tabs(["Data Exploration", "Model Training & Evaluation", "Real-time Prediction"])

with tab1:
    st.header("Data Exploration")
    
    # Data upload or sample data selection
    data_option = st.radio("Select data source:", ["Use Sample Data", "Upload Your Own Data"])
    
    if data_option == "Upload Your Own Data":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("Data successfully loaded!")
        else:
            st.info("Please upload a CSV file or select 'Use Sample Data'")
            data = None
    else:
        data = load_sample_data()
        st.success("Sample data loaded!")
    
    if data is not None:
        # Display data overview
        st.subheader("Data Overview")
        st.write(f"Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")
        st.dataframe(data.head())
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(data.describe())
        
        # Class distribution
        st.subheader("Class Distribution")
        class_counts = data['Class'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
        ax.set_xlabel('Class (0: Normal, 1: Fraud)')
        ax.set_ylabel('Count')
        ax.set_title('Transaction Class Distribution')
        
        # Add percentage labels
        total = len(data)
        for i, count in enumerate(class_counts.values):
            percentage = count / total * 100
            ax.text(i, count + 5, f"{percentage:.1f}%", ha='center')
            
        st.pyplot(fig)
        
        # Feature correlations
        st.subheader("Feature Correlations")
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        st.pyplot(fig)
        
        # Feature distributions by class
        st.subheader("Feature Distributions by Class")
        feature_to_plot = st.selectbox("Select feature to visualize:", data.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x=feature_to_plot, hue='Class', kde=True, ax=ax)
        st.pyplot(fig)

with tab2:
    st.header("Model Training & Evaluation")
    
    if data is not None:
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            C = st.slider("Regularization strength (C):", 0.01, 10.0, 1.0, 0.01)
            class_weight_option = st.radio("Class weights:", ["None", "Balanced"])
            class_weight = "balanced" if class_weight_option == "Balanced" else None
        
        with col2:
            st.write("**Parameter Explanation:**")
            st.write("**Regularization (C):** Higher values reduce regularization. Use lower values to prevent overfitting.")
            st.write("**Class weights:** Use 'Balanced' for imbalanced datasets to give more importance to the minority class.")
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Preprocess data
                X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, X_train, X_test = preprocess_data(data)
                
                # Train model
                model = train_model(X_train_scaled, y_train, C=C, class_weight=class_weight)
                
                # Evaluate model
                cm, fpr, tpr, roc_auc, precision, recall, report, y_prob = evaluate_model(model, X_test_scaled, y_test)
                
                # Plot evaluation metrics
                plot_evaluation(cm, fpr, tpr, roc_auc, precision, recall, report)
                
                # Model explainability
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
                X_test_df['Class'] = y_test.values
                explain_model(model, X_test_scaled, X_test_df, feature_names)
                
                # Save model to session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = feature_names
                
                # Download model
                st.markdown(download_model(model), unsafe_allow_html=True)

with tab3:
    st.header("Real-time Prediction")
    
    if data is not None:
        if 'model' in st.session_state:
            st.success("Model is ready for predictions!")
            
            # Create input form for new transaction
            st.subheader("Enter Transaction Details")
            
            # Get feature names excluding 'Class'
            features = [col for col in data.columns if col != 'Class']
            
            # Create input fields for each feature
            input_data = {}
            
            # Create two columns for input fields
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(features):
                # Alternate between columns
                with col1 if i % 2 == 0 else col2:
                    # Get mean and std for the feature to provide a reasonable default
                    mean_val = data[feature].mean()
                    std_val = data[feature].std()
                    
                    # Create a slider with a reasonable range based on the data
                    input_data[feature] = st.number_input(
                        f"{feature}:",
                        value=float(mean_val),
                        step=float(std_val/10),
                        format="%.2f"
                    )
            
            # Make prediction
            if st.button("Predict"):
                # Create DataFrame from input
                input_df = pd.DataFrame([input_data])
                
                # Scale the input
                input_scaled = st.session_state['scaler'].transform(input_df)
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_scaled)[0]
                probability = st.session_state['model'].predict_proba(input_scaled)[0][1]
                
                # Display result
                if prediction == 0:
                    st.success(f"Transaction appears to be LEGITIMATE (Fraud Probability: {probability:.2%})")
                else:
                    st.error(f"Transaction appears to be FRAUDULENT (Fraud Probability: {probability:.2%})")
                
                # Explain the prediction
                st.subheader("Prediction Explanation")
                
                # Calculate feature contributions for this prediction
                features_df = pd.DataFrame({
                    'Feature': st.session_state['feature_names'],
                    'Value': input_scaled[0],
                    'Coefficient': st.session_state['model'].coef_[0],
                    'Original_Value': input_df.values[0]
                })
                
                features_df['Contribution'] = features_df['Value'] * features_df['Coefficient']
                features_df['Abs_Contribution'] = abs(features_df['Contribution'])
                features_df = features_df.sort_values('Abs_Contribution', ascending=False)
                
                # Show top 5 contributing features
                st.subheader("Top Contributing Features")
                top_features = features_df.head(5)
                
                # Create a horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['red' if x < 0 else 'green' for x in top_features['Contribution']]
                sns.barplot(x='Abs_Contribution', y='Feature', data=top_features, palette=colors, ax=ax)
                ax.set_title('Top 5 Features Influencing Prediction')
                ax.set_xlabel('Contribution Magnitude')
                st.pyplot(fig)
                
                # Show feature values table with contribution
                st.subheader("Feature Contributions")
                contribution_df = features_df[['Feature', 'Original_Value', 'Contribution']]
                contribution_df.columns = ['Feature', 'Value', 'Contribution to Prediction']
                st.dataframe(contribution_df.style.format({"Value": "{:.2f}", "Contribution to Prediction": "{:.4f}"}))
                
        else:
            st.warning("Please train a model in the 'Model Training & Evaluation' tab first.")
    else:
        st.warning("Please load data in the 'Data Exploration' tab first.")

# Footer
st.markdown("---")
st.markdown("Credit Card Fraud Detection System | Built with Streamlit and Scikit-learn")