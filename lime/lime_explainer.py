from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import joblib

# Load the scaler
scaler = joblib.load('model/scaler/minmax_scaler.pkl')
# Function to explain predictions using LIME
def explain_prediction(input_features):
    # Load the model
    model = load_model_file("model/cnn_model.h5")
    
    # Create a LIME explainer for tabular data
    explainer = LimeTabularExplainer(
        training_data=scaler.inverse_transform(np.random.rand(100, 10)),  # Dummy data to initialize LIME with shape of input
        mode='classification', 
        training_labels=None,  # If you have labels, pass them here
        feature_names=["Al", "Sc", "Hemo", "PCV", "RBCC", "DM", "SU", "BGR", "HTN", "Appet"],  # Add feature names
        class_names=["CKD", "Healthy"],  # Output class names
        discretize_continuous=True
    )
    
    # Wrap the model's predict function for LIME
    def wrapped_predict(input_data):
        input_data_transformed = scaler.transform(input_data)
        reshaped_input_data = input_data_transformed.reshape(input_data_transformed.shape[0], input_data_transformed.shape[1], 1)
        prediction_prob = model.predict(reshaped_input_data)
        return prediction_prob

    # Generate explanation for a specific prediction
    explanation = explainer.explain_instance(
        input_features=input_features, 
        predict_fn=wrapped_predict,  # Pass the wrapped predict function to LIME
        num_features=10  # Number of features to display in explanation
    )

    # Return the explanation object
    return explanation

# Example usage
input_features = [0.6, 1.2, 14.0, 35.0, 1.5, 0, 0, 120, 0, 1]  # Example input features (make sure to replace with actual values)
explanation = explain_prediction(input_features)

# Display the explanation
explanation.show_in_notebook()
