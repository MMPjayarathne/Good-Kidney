from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import joblib
from model.model_service import load_model_file

# Load the previously used scaler
scaler = joblib.load('model/scaler/scaler.pkl')

# Function to explain predictions using LIME
def explain_prediction_lime(input_data):
    input_features = np.array(input_data).reshape(1, -1)
    model = load_model_file("model/cnn_model.h5")
    feature_labels = [
        "Specific Gravity", "Albumin", "Serum Creatinine", "Hemoglobin", 
        "Packed Cell Volume", "Red Blood Cell Count", "Diabetes Mellitus", 
        "Blood Glucose Random", "Hypertension", "Appetite"
    ]
    train_df = pd.read_csv('explainer/train_data.csv')
    X_train = train_df.drop('class', axis=1)
    
    explainer = LimeTabularExplainer(
        training_data=scaler.inverse_transform(X_train), 
        mode='classification', 
        training_labels=None,
        feature_names=feature_labels,
        class_names=["CKD", "Healthy"], 
        discretize_continuous=True
    )
    
    def wrapped_predict(input_data):
        input_data = np.array(input_data)
        scaled_input_data = scaler.transform(input_data)
        reshaped_input_data = scaled_input_data.reshape(scaled_input_data.shape[0], scaled_input_data.shape[1], 1)
        prediction_prob = model.predict(reshaped_input_data)
        if prediction_prob.shape[1] == 1:
            return np.hstack([1 - prediction_prob, prediction_prob])
        return prediction_prob

    explanation = explainer.explain_instance(
        data_row=input_features[0],
        predict_fn=wrapped_predict,  
        num_features=10,
    )
    
    explanation_list = explanation.as_list()
    
    def clean_feature_name(feature):
        # Remove any numerical values or comparison operators around feature names
        tokens = feature.split()
        cleaned_tokens = [token for token in tokens if not any(char.isdigit() or char in ['<', '<=', '>', '>='] for char in token)]
        return " ".join(cleaned_tokens)
    
    explanation_text = "\n".join([
        f"{clean_feature_name(feature)}: "
        f"{'A positive contribution' if weight > 0 else 'A negative contribution'} ({abs(weight):.3f}) "
        f"{'suggests an increased likelihood of CKD.' if weight > 0 else 'indicates a reduced likelihood of CKD.'}"
        for feature, weight in explanation_list
    ])

    return explanation_text
