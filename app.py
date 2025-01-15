import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from model.model_service import load_model_file, predict
from views.plotting import *
from explainer.lime_explainer import explain_prediction_lime
from explainer.shap_explainer import explain_prediction_shap
from bioT.medical_explanation import MedicalExplanationGenerator

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
        # Convert categorical features to numeric (ensure they're integers)
        dm = 1 if dm == "Yes" else 0  
        htn = 1 if htn == "Yes" else 0 
        appet = 0 if appet == "Good" else 1 
        
        # Prepare input for the model (log1p transformation where needed)
        input_features = [
            al, 
            np.log1p(sc),  
            np.log1p(hemo),  
            np.log1p(pcv),  
            np.log1p(rbcc), 
            dm,  
            su,  
            np.log1p(bgr),  
            htn,  
            appet 
        ]

        # Prepare input array for the model with 1 sample and 10 features
        input_data = np.array([input_features])
        
        # Get prediction probability from the model
        prediction_prob = predict(model, input_data)
        
        # Convert probabilities to class predictions (0 or 1)
        prediction_class = (prediction_prob >= 0.5).astype(int)  # Class prediction (0 or 1)
        prediction = prediction_class[0][0]
        probability_healthy = (1 - prediction_prob[0][0]) * 100  # Convert to percentage
        probability_ckd = prediction_prob[0][0] * 100  # Convert to percentage

        # Display the prediction with color-coded messages
        if prediction == 1:  # Prediction is CKD
            st.warning(f"The patient is likely to have Chronic Kidney Disease ")
            st.warning(f"The probability of having Chronic Kidney Disease is: {probability_ckd:.2f}%")
        else:  # Prediction is Healthy
            st.success(f"The patient is Healthy")
            st.success(f"The probability of being Healthy is: {probability_healthy:.2f}%")
        
        # Pie chart showing overall prediction breakdown
        
        st.sidebar.title("Analysis")
        plotPieChart(prediction,probability_ckd,probability_healthy)

        explanation_lime = explain_prediction_lime(input_data)
        st.title("Explanation of the model's prediction using LIME")
        st.text(explanation_lime)
        
        explanation_shap = explain_prediction_shap(input_data)
        
        st.title("Explanation of the model's prediction using SHAP & BioBERT")
        # st.text("SHAP Explanation:\n", explanation_shap["shap_explanation"])
        # st.text("Medical Guidance:\n", explanation_shap["medical_guidance"])
        
        feature_values = {
        "Albumin": al,
        "Serum Creatinine": sc,
        "Hemoglobin": hemo,
        "Packed Cell Volume": pcv,
        "Red Blood Cell Count": rbcc,
        "Diabetes Mellitus": dm,
        "Sugar": su,
        "Blood Glucose Random": bgr,
        "Hypertension": htn,
        "Appetite": appet
        }
       
        
        print(f"SHAP values:",explanation_shap)
        print(f"feature Values: ",feature_values)
        # Check SHAP values type
        print("SHAP values types:")
        for val in explanation_shap:
            print(f"{val}: {type(val)}")
        
        # Check feature values types
        print("Feature values types:")
        for feature, value in feature_values.items():
            print(f"{feature}: {value} ({type(value)})")
            
        # Check lengths
        print(f"Number of feature labels: {len(feature_labels)}")
        print(f"Number of SHAP values: {len(explanation_shap)}")
        generator = MedicalExplanationGenerator()
        
        # features = list(shap_values.keys())
        # shap_vals = [shap_values[feature] for feature in features]
        # aligned_feature_values = {feature: feature_values[feature] for feature in features}

        # # Generate explanations
        # result = generator.explain_prediction(features, shap_vals, aligned_feature_values)
        result = generator.explain_prediction(
        feature_labels,
        explanation_shap,
        feature_values
        )
        
        st.sidebar.title("\nSHAP Explanation:")
        st.sidebar.write(result["shap_explanation"])
        
        st.title("\nMedical Insights:")
        st.write(result["medical_insights"])
        
        st.title("\nPrecautions:")
        for precaution in result["precautions"]:
            st.write(f"- {precaution}")
        
        st.title("\nFeature Units:")
        for feature, unit in result["feature_units"].items():
            st.write(f"{feature}: {unit}")


    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(e)

# Sidebar for extra information
st.sidebar.title("About")
st.sidebar.write("This app predicts outputs based on the inputs provided using a trained model.")
