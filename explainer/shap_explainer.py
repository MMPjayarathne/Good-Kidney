import shap
import numpy as np
import joblib
from bioT.medical_guidance import explain_prediction_with_gpt
from model.model_service import load_model_file
import pandas as pd
import streamlit as st

# Load the scaler
scaler = joblib.load('model/scaler/scaler.pkl')

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
        model = load_model_file("model/cnn_model.h5")

        # Feature labels for the dataset
        feature_labels = [
            "Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume", 
            "Red Blood Cell Count", "Diabetes Mellitus", "Sugar", "Blood Glucose Random", 
            "Hypertension", "Appetite"
        ]
        
        # Scale the input data using the pre-trained scaler
        scaled_input_data = scaler.transform(input_features)
        # print("Scaled input data:", scaled_input_data)

        # Ensure SHAP input matches the expected format
        flattened_input = scaled_input_data.flatten().reshape(1, -1)
        # print("Flattened input shape:", flattened_input.shape)

        # Define a wrapper for model prediction, reshaping for the model input
        def model_predict(input_data):
            reshaped_input = np.array(input_data).reshape(-1, 10, 1)
            prediction_prob = model.predict(reshaped_input)
            # print("Model prediction probabilities:", prediction_prob)
            return prediction_prob
        
        # Create a SHAP explainer using the training data and prediction function
        explainer = shap.KernelExplainer(model_predict, X_train.values)
        # print("SHAP explainer initialized")
        
        # Explain the prediction for the input data
        shap_values = explainer.shap_values(flattened_input)
        # print("SHAP values calculated:", shap_values)
        # print("flattened_input:", flattened_input)

        # Reshape SHAP values if needed
        # Assuming shap_values is a list with one array per output class
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values = shap_values[0]  # Use the SHAP values for the first output
            # print("Processed SHAP values shape:", shap_values.shape)

        # Reshape SHAP values to match the flattened input if necessary
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)  # Remove the last dimension
            # print("Reshaped SHAP values shape:", shap_values.shape)

        # Validate the shapes
        # print("Final SHAP values shape:", shap_values.shape)
        # print("Flattened input shape:", flattened_input.shape)

        # Visualize the SHAP summary plot
        shap.summary_plot(
            shap_values, flattened_input, feature_names=feature_labels
        )
        # print("SHAP summary plot displayed")


        # Generate textual explanation of SHAP values
        if shap_values.ndim > 1:
            shap_values = shap_values.flatten()  # Flatten if multi-dimensional
            # print("Flattened SHAP values for explanation:", shap_values)

        # Validate the length of shap_values against feature_labels
        if len(shap_values) != len(feature_labels):
            raise ValueError(
                f"Mismatch between SHAP values ({len(shap_values)}) and feature labels ({len(feature_labels)})."
            )

        # Create explanation text
        explanation_text = "\n".join([
            f"{feature}: {shap_value:.3f}"
            for feature, shap_value in zip(feature_labels, shap_values)
        ])
        # print("Generated explanation text:", explanation_text)
        # st.sidebar.title("\nSHAP Explanation:")
        # st.sidebar.write(explanation_text)
        # print(f"SHAP values: ",explanation_text)
        return shap_values


    except Exception as e:
        print("An error occurred:", e)
        raise

