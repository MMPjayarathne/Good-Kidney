import shap
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st

# Load the scaler
scaler = joblib.load('model/scaler/scaler.pkl')

# Function to load the model
def load_model_file(model_path="model/mlp_model.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    return model

# Function to explain predictions using SHAP
def explain_prediction_shap(input_data):
    try:
        # Convert input_features to a NumPy array and reshape to (1, num_features)
        input_features = np.array(input_data).reshape(1, -1)

        # Reload the training dataset
        train_df = pd.read_csv('explainer/train_data.csv')
        X_train = train_df.drop('class', axis=1)
        y_train = train_df['class']
    
        # Load the trained model
        model = load_model_file("model/mlp_model.h5")

        # Feature labels for the dataset
        feature_labels = [
            "Specific Gravity", "Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume", 
            "Red Blood Cell Count", "Diabetes Mellitus", "Blood Glucose Random", 
            "Hypertension", "Appetite"
        ]
        
        # Scale the input data using the pre-trained scaler
        scaled_input_data = scaler.transform(input_features)
        # print("Scaled input data shape:", scaled_input_data.shape)

        # Define a wrapper for model prediction
        def model_predict(input_data):
            # Ensure the input data is in the correct shape
            reshaped_input = np.array(input_data).reshape(-1, len(feature_labels))
            scaled_input = scaler.transform(reshaped_input)
            prediction_prob = model.predict(scaled_input)
            print("Model prediction probabilities shape:", prediction_prob.shape)
            return prediction_prob
        
        # Create a SHAP explainer using a subset of the training data (for efficiency)
        background_data = shap.sample(X_train.values, 100)  # Use 100 samples as background
        explainer = shap.KernelExplainer(model_predict, background_data)
        print("SHAP explainer initialized")
        
        # Explain the prediction for the input data
        shap_values = explainer.shap_values(scaled_input_data)
        print("SHAP values shape:", np.array(shap_values).shape)
        # print("SHAP values :", shap_values)

        # Extract SHAP values for the positive class (class 1)
        if isinstance(shap_values, list):
            # For binary classification, shap_values is a list of two arrays (one for each class)
            # We use the SHAP values for the positive class (class 1)
            shap_values_class1 = np.array(shap_values[1])  # Get SHAP values for class 1
            # print("SHAP values for class 1 shape:", shap_values_class1.shape)
        else:
            # For single-output models, shap_values is a single array
            shap_values_class1 = np.array(shap_values)
            # print("SHAP values for single output shape:", shap_values_class1.shape)

        # The shape is (1, 10, 2), meaning you have 2 SHAP values for each feature. You need to select the second one (class 1)
        shap_values_class1 = shap_values_class1[0, :, 1]  # Select the SHAP values for class 1 (second index)
        # print("SHAP values for class 1:", shap_values_class1)

        # Flatten the SHAP values to match the number of features
        shap_values_class1 = shap_values_class1.flatten()
        # print("Flattened SHAP values shape:", shap_values_class1.shape)

        # Validate the length of shap_values against feature_labels
        if len(shap_values_class1) != len(feature_labels):
            raise ValueError(
                f"Mismatch between SHAP values ({len(shap_values_class1)}) and feature labels ({len(feature_labels)})."
            )

        # Create explanation text
        explanation_text = "\n".join([
            f"{feature}: {shap_value:.3f}"
            for feature, shap_value in zip(feature_labels, shap_values_class1)
        ])
        # print("Generated explanation text:", explanation_text)
        # st.sidebar.title("SHAP Explanation:")
        # st.sidebar.write(explanation_text)

        # print(f"SHAP values: ", explanation_text)
        return shap_values_class1

    except Exception as e:
        print("An error occurred:", e)
        raise