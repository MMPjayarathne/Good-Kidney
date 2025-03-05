import numpy as np
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the scaler
scaler = joblib.load('model/scaler/scaler.pkl')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Function to load the model
def load_model_file(model_path="model/mlp_model.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    return model

# Function to make predictions
def predict(model, input_data):
    # Print the raw input data for debugging
    print(f"Raw Input Data: {input_data}")
    
    # Apply MinMaxScaler to scale the features as during training
    scaled_input_data = scaler.transform(input_data)  # No transpose here, just pass the data as is
    
    prediction_prob = model.predict(scaled_input_data)  # Get the probabilities for class 1

    return prediction_prob
