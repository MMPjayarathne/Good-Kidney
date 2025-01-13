import numpy as np
from tensorflow.keras.models import load_model
import os

# Function to load the model
def load_model_file(model_path="model/cnn_model.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    return model

# Function to make predictions
def predict(model, input_features):
    input_data = np.array([input_features])
    prediction = model.predict(input_data)
    return prediction[0][0]
