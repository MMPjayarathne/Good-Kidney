from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import joblib
from model.model_service import load_model_file

# Same scaler used previously 
scaler = joblib.load('model/scaler/scaler.pkl')

# Function to explain predictions using LIME
def explain_prediction_lime(input_data):
    
    input_features = np.array(input_data).reshape(1, -1)
    model = load_model_file("model/cnn_model.h5")
    feature_labels = [
        "Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume", 
        "Red Blood Cell Count", "Diabetes Mellitus", "Sugar", "Blood Glucose Random", 
        "Hypertension", "Appetite"
    ]
    train_df = pd.read_csv('explainer/train_data.csv')
    # print("Training data loaded:", train_df.shape)
    X_train = train_df.drop('class', axis=1)
    
    # Create a LIME explainer for tabular data
    explainer = LimeTabularExplainer(
        training_data=scaler.inverse_transform(X_train), 
        mode='classification', 
        training_labels=None,
        feature_names=feature_labels,
        class_names=["CKD", "Healthy"],  # Output class names
        discretize_continuous=True
    )
    
    # In the wrapped_predict function, adjust the output to return two probabilities (one for each class)
    def wrapped_predict(input_data):
        
        input_data = np.array(input_data)
        scaled_input_data = scaler.transform(input_data)
        reshaped_input_data = scaled_input_data.reshape(scaled_input_data.shape[0], scaled_input_data.shape[1], 1)
        prediction_prob = model.predict(reshaped_input_data)
        
        if prediction_prob.shape[1] == 1:  # If the model returns one probability (binary classification)
            # Manually create a 2-class probability distribution
            return np.hstack([1 - prediction_prob, prediction_prob])  # [1 - prob_class_1, prob_class_1]
        
        return prediction_prob

    # Generate explanation for a specific prediction (around class 1 - CKD)
    explanation = explainer.explain_instance(
        data_row=input_features[0],
        predict_fn=wrapped_predict,  # Passing the wrapped predict function to LIME
        num_features=10,
        
    )
    
    explanation_list = explanation.as_list()  # Fetch explanation for class 1 (CKD)

    # Convert the explanation list into a more readable format
    explanation_text = "\n".join([f"{feature}: {weight:.3f}" for feature, weight in explanation_list])

    # Return the explanation object
    return explanation_text
