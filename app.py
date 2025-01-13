import streamlit as st
import matplotlib.pyplot as plt
from model.model_service import load_model_file, predict
from views.plotting import *

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def get_model():
    return load_model_file()

# Load the model
model = get_model()

# Frontend layout
st.title("GOOD-KIDNEY")
st.markdown("This application will predict the presence of Chronic Kidney Disease using Deep Learning. Enter the features below to get predictions:")

# Divide input fields into multiple rows
feature_labels = [
    "Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume ", 
    "Red Blood Cell Count", "Diabetes Mellitus", "Sugar", "Blood Glucose Random", 
    "Hypertension", "Appetite"
]

# Albumin options
al_options = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
su_options = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# First row
col1, col2, col3 = st.columns(3)
with col1:
    al = st.selectbox("Albumin", options=al_options, index=0)
with col2:
    sc = st.number_input("Serum Creatinine (mgs/dl)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
with col3:
    hemo = st.number_input("Hemoglobin (gms)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

# Second row
col4, col5, col6 = st.columns(3)
with col4:
    pcv = st.number_input("Packed Cell Volume (PCV) (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
with col5:
    rbcc = st.number_input("Red Blood Cell Count (RBCC) (mill/cmm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
with col6:
    dm = st.selectbox("Diabetes Mellitus", options=["Yes", "No"], index=0)

# Third row
col7, col8, col9 = st.columns(3)
with col7:
    su = st.selectbox("Sugar", options=su_options, index=0)
with col8:
    bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
with col9:
    htn = st.selectbox("Hypertension", options=["Yes", "No"], index=0)

# Fourth row for Appetite
col10, _ = st.columns([1, 2])  # Adjust column width for balance
with col10:
    appet = st.selectbox("Appetite", options=["Good", "Poor"], index=0)
    
    
# Prediction button
if st.button("Predict"):
    try:
        # Prepare input for the model (convert categorical features to numeric)
        input_features = [
            al, sc, hemo, pcv, rbcc, 
            1 if dm == "Yes" else 0,  # Convert 'Yes'/'No' to 1/0
            su, bgr, 
            1 if htn == "Yes" else 0,  # Convert 'Yes'/'No' to 1/0
            0 if appet == "Good" else 1  # Convert 'Good'/'Poor' to 0/1
        ]

        # Get prediction probability from the model
        prediction_prob = predict(model, input_features)
        
        # Convert probabilities to class predictions (0 or 1)
        prediction_class = (prediction_prob >= 0.5).astype(int)  # Class prediction (0 or 1)
        prediction = prediction_class[0][0]
        probability_ckd = (1 - prediction_prob[0][0]) * 100  # Convert to percentage
        probability_healthy = prediction_prob[0][0] * 100  # Convert to percentage

        # Display the prediction with color-coded messages
        if prediction == 0:  # Prediction is CKD
            st.warning(f"The patient is likely to have Chronic Kidney Disease ")
            st.warning(f"The probability of having Chronic Kidney Disease is: {probability_ckd:.2f}%")
        else:  # Prediction is Healthy
            st.success(f"The patient is Healthy")
            st.success(f"The probability of being Healthy is: {probability_healthy:.2f}%")
        
        # Pie chart showing overall prediction breakdown
        
        st.sidebar.title("Analysis")
        plotPieChart(prediction,probability_ckd,probability_healthy)
       

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar for extra information
st.sidebar.title("About")
st.sidebar.write("This app predicts outputs based on the inputs provided using a trained model.")
