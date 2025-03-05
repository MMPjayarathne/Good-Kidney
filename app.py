import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from model.model_service import load_model_file, predict
from views.plotting import *
from views.loading import start_loading,stop_loading
from explainer.lime_explainer import explain_prediction_lime
from explainer.shap_explainer import explain_prediction_shap
from medical_guidance.medical_explanation import MedicalExplanationGenerator
from medical_guidance.medical_explanation_gemini import explain_prediction_with_gemini
import time
import joblib

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def get_model():
    return load_model_file()

# Load the model
model = get_model()




# Add a logo
st.sidebar.image("assests/good kidney.png", width=250) 
st.title("Welcome! 🧑‍⚕️")
st.markdown("This application will predict the presence of Chronic Kidney Disease using Deep Learning.🕵️ Enter the features below to get predictions:")

# Divide input fields into multiple rows
feature_labels = [
    "Specific Gravity","Albumin", "Serum Creatinine", "Hemoglobin", "Packed Cell Volume ", 
    "Red Blood Cell Count", "Diabetes Mellitus", "Blood Glucose Random", 
    "Hypertension", "Appetite"
]

# Albumin options
al_options = [0, 1, 2, 3, 4, 5]
sg_options = [1.005, 1.01, 1.015, 1.02, 1.025]
sg_mapping = {'1.005': 0, '1.01': 1, '1.015': 2, '1.02': 3, '1.025': 4}

# Function to map an sg value
def map_sg_value(sg_value):
    sg_str = str(sg_value)  
    return sg_mapping.get(sg_str, None)

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
    pcv = st.number_input("Packed Cell Volume (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
with col5:
    rbcc = st.number_input("Red Blood Cell Count (mill/cmm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
with col6:
    dm = st.selectbox("Diabetes Mellitus", options=["Yes", "No"], index=0)

# Third row
col7, col8, col9 = st.columns(3)
with col7:
    sg = st.selectbox("Specific Gravity", options=sg_options, index=0)
with col8:
    bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=0.0, max_value=400.0, value=0.0, step=0.1)
with col9:
    htn = st.selectbox("Hypertension", options=["Yes", "No"], index=0)

# Fourth row for Appetite
col10, _ = st.columns([1, 2])  # Adjust column width for balance
with col10:
    appet = st.selectbox("Appetite", options=["Good", "Poor"], index=0)
    
    
# Prediction button
if st.button("Predict"):
    # start_loading()
    try:
        # Convert categorical features to numeric (ensure they're integers)
        dm = 1 if dm == "Yes" else 0  
        htn = 1 if htn == "Yes" else 0 
        appet = 0 if appet == "Good" else 1 
        
        sg = map_sg_value(sg)
        
        # Prepare input for the model (log1p transformation where needed)
        input_features = [
            sg,
            al, 
            np.log1p(sc),  
            np.log1p(hemo),  
            np.log1p(pcv),  
            np.log1p(rbcc), 
            dm,    
            np.log1p(bgr),  
            htn,  
            appet 
        ]

        # Prepare input array for the model with 1 sample and 10 features
        input_data = np.array([input_features])
        
        # Get prediction probabilities from the model
        prediction_prob = predict(model, input_data)

        # Convert probabilities to class predictions (0 or 1)
        prediction = np.argmax(prediction_prob, axis=1)[0]  # Extract scalar value

        # Extract probabilities correctly
        probability_ckd = prediction_prob[0][1] * 100  # Probability of CKD (assuming class 1 is CKD)
        probability_healthy = prediction_prob[0][0] * 100  # Probability of being healthy (assuming class 0 is healthy)

        # Display results based on the class prediction
        if prediction == 1:
            st.warning("⚠️ The patient is likely to have Chronic Kidney Disease")
            st.warning(f"The probability of having Chronic Kidney Disease is: {probability_ckd:.2f}%")
        else:
            st.success("🥦 The patient is Healthy")
            st.success(f"The probability of being Healthy is: {probability_healthy:.2f}%")


        # Pie chart showing overall prediction breakdown
        
        st.sidebar.title("Analysis 🔬")
        plotPieChart(prediction,probability_ckd,probability_healthy)

        st.sidebar.title("LIME Explanation")
        explanation_lime = explain_prediction_lime(input_data)
    
        # st.sidebar.text(visualize_lime_explanation_from_text(explanation_lime))
        st.sidebar.text(explanation_lime)
        
        explanation_shap = explain_prediction_shap(input_data)
        
        st.title("Explanation of the model's prediction using SHAP & Gemini")
        # st.text("SHAP Explanation:\n", explanation_shap["shap_explanation"])
        # st.text("Medical Guidance:\n", explanation_shap["medical_guidance"])
        
        feature_values = {
        "Specific Gravity" :sg,
        "Albumin": al,
        "Serum Creatinine": sc,
        "Hemoglobin": hemo,
        "Packed Cell Volume": pcv,
        "Red Blood Cell Count": rbcc,
        "Diabetes Mellitus": dm,
        "Blood Glucose Random": bgr,
        "Hypertension": htn,
        "Appetite": appet
        }
       
        
        # print(f"SHAP values:",explanation_shap)
        # print(f"feature Values: ",feature_values)
        # # Check SHAP values type
        # print("SHAP values types:")
        # for val in explanation_shap:
        #     print(f"{val}: {type(val)}")
        
        # # Check feature values types
        # print("Feature values types:")
        # for feature, value in feature_values.items():
        #     print(f"{feature}: {value} ({type(value)})")
            
        # # Check lengths
        # print(f"Number of feature labels: {len(feature_labels)}")
        # print(f"Number of SHAP values: {len(explanation_shap)}")
        # generator = MedicalExplanationGenerator()
        
        # features = list(shap_values.keys())
        # shap_vals = [shap_values[feature] for feature in features]
        # aligned_feature_values = {feature: feature_values[feature] for feature in features}

        # # Generate explanations
        # result = generator.explain_prediction(features, shap_vals, aligned_feature_values)
        # result = generator.explain_prediction(
        # feature_labels,
        # explanation_shap,
        # feature_values
        # )
    
        
        result = explain_prediction_with_gemini(feature_labels,explanation_shap,feature_values,prediction)
        
        st.sidebar.title("\nSHAP Explanation:")
        st.sidebar.write(result["shap_explanation"])
        
        
        st.title("\nMedical Insights:")
        st.write(result["medical_guidance"])
        # stop_loading()
        
        
        # st.title("\nPrecautions:")
        # for precaution in result["precautions"]:
        #     st.write(f"- {precaution}")
        
        # st.title("\nFeature Units:")
        # for feature, unit in result["feature_units"].items():
        #     st.write(f"{feature}: {unit}")


    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(e)

# Sidebar for extra information
st.sidebar.title("About")
st.sidebar.write("This app predicts outputs based on the inputs provided using a trained model.")



