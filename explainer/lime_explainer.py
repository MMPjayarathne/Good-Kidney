from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import joblib
from model.model_service import load_model_file

# Load the scaler
scaler = joblib.load('model/scaler/scaler.pkl')

# Function to explain predictions using LIME
def explain_prediction_lime(input_data):
    # Convert input_features to a NumPy array (in case it's a list)
    input_features = np.array(input_data).reshape(1, -1)  # Ensure the shape is (1, num_features)

    # Load the model
    model = load_model_file("model/cnn_model.h5")
    feature_labels = [
        "Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume", 
        "Red Blood Cell Count", "Diabetes Mellitus", "Sugar", "Blood Glucose Random", 
        "Hypertension", "Appetite"
    ]
    
    # Create a LIME explainer for tabular data
    explainer = LimeTabularExplainer(
        training_data=scaler.inverse_transform(np.random.rand(100, 10)),  # Dummy data to initialize LIME with shape of input
        mode='classification', 
        training_labels=None,  # If you have labels, pass them here
        feature_names=feature_labels,
        class_names=["CKD", "Healthy"],  # Output class names
        discretize_continuous=True
    )
    
    # In the wrapped_predict function, adjust the output to return two probabilities (one for each class)
    def wrapped_predict(input_data):
        # Ensure input_data is a NumPy array
        input_data = np.array(input_data)
        
        # Apply the scaler
        scaled_input_data = scaler.transform(input_data)
        
        # Reshape input data for the model
        reshaped_input_data = scaled_input_data.reshape(scaled_input_data.shape[0], scaled_input_data.shape[1], 1)
        
        # Predict using the model and return probabilities
        prediction_prob = model.predict(reshaped_input_data)
        
        # If it's a binary classification, convert it to two probabilities (e.g., for class 0 and class 1)
        if prediction_prob.shape[1] == 1:  # If the model returns one probability (binary classification)
            # Manually create a 2-class probability distribution
            return np.hstack([1 - prediction_prob, prediction_prob])  # [1 - prob_class_1, prob_class_1]
        
        return prediction_prob

    # Generate explanation for a specific prediction (around class 1 - CKD)
    explanation = explainer.explain_instance(
        data_row=input_features[0],  # Pass the input instance for which you want to explain (first row of input_features)
        predict_fn=wrapped_predict,  # Pass the wrapped predict function to LIME
        num_features=10,  # Number of features to display in explanation
        
    )
    
    # Extract explanation as a list of feature importance
    explanation_list = explanation.as_list()  # Fetch explanation for class 1 (CKD)

    # Convert the explanation list into a more readable format
    explanation_text = "\n".join([f"{feature}: {weight:.3f}" for feature, weight in explanation_list])

    # Return the explanation object
    return explanation_text
