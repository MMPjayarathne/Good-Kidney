import numpy as np
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the scaler
scaler = joblib.load('model/scaler/minmax_scaler.pkl')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Function to load the model
def load_model_file(model_path="model/cnn_model.h5"):
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
    # Print the scaled input data for debugging
    # print(f"Scaled Input Data: {scaled_input_data}")
    
    # Reshape the input data to match the shape used during training (samples, features, 1)
    reshaped_input_data = scaled_input_data.reshape(scaled_input_data.shape[0], scaled_input_data.shape[1], 1)
    # print(f"Ready to predict Data: {reshaped_input_data}")
    # Predict using the model
    prediction_prob = model.predict(reshaped_input_data)  # Get the probabilities for class 1
    # print(f"Prediction : {prediction_prob}")

    return prediction_prob
