import shap
import numpy as np
import joblib
from model.model_service import load_model_file

# Load the scaler
scaler = joblib.load('model/scaler/minmax_scaler.pkl')

# Function to explain predictions using SHAP
def explain_prediction_shap(input_data):
    # Convert input_features to a NumPy array (in case it's a list)
    input_features = np.array(input_data).reshape(1, -1)  # Ensure the shape is (1, num_features)

    # Load the model
    model = load_model_file("model/cnn_model.h5")

    # Feature labels for your dataset
    feature_labels = [
        "Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume", 
        "Red Blood Cell Count", "Diabetes Mellitus", "Sugar", "Blood Glucose Random", 
        "Hypertension", "Appetite"
    ]
    
    # Apply the scaler to the input data
    scaled_input_data = scaler.transform(input_features)
    
    # Define a wrapper for model prediction, making sure the output is in the right format
    def model_predict(input_data):
        # Ensure the input data is in the right shape (e.g., (1, num_features))
        input_data = np.array(input_data).reshape(1, -1)
        # Predict using the model and return the probabilities (should be 2 probabilities for binary classification)
        prediction_prob = model.predict(input_data)
        return prediction_prob  # For binary classification, this should return an array of shape (1, 2)
    
    # Create a SHAP explainer
    explainer = shap.KernelExplainer(model_predict, scaled_input_data)
    
    # Explain the model's prediction for the input data
    shap_values = explainer.shap_values(scaled_input_data)

    # Visualize the explanation as a summary plot
    shap.initjs()
    shap.summary_plot(shap_values, scaled_input_data, feature_names=feature_labels)

    # For returning a textual explanation
    explanation_text = "\n".join([f"{feature}: {shap_value.mean():.3f}" 
                                 for feature, shap_value in zip(feature_labels, shap_values[0])])

    return explanation_text

