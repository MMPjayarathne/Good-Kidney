import streamlit as st
import matplotlib.pyplot as plt
from model.model_service import load_model_file, predict

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def get_model():
    return load_model_file()

# Load the model
model = get_model()

# Frontend layout
st.title("GOOD-KIDNEY")
st.markdown("This application will predict the present of Chronic Kidney Disease using Deep Learning. Enter the features below to get predictions:")

# Input fields for features
sg, al, sc, dm, htn, appet = st.columns(6)

#the specific gravity options
sg_options = [1.005, 1.010, 1.015, 1.020, 1.025, 1.030]
al_options = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

with sg:
    sg = st.selectbox("Specific Gravity", options=sg_options, index=0)
with al:
    al = st.selectbox("Albumin", options=al_options, index=0)
with sc:
    sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
with dm:
    dm = st.selectbox("Diabetes Mellitus", options=["Yes", "No"], index=0)
with htn:
    htn = st.selectbox("Hypertension", options=["Yes", "No"], index=0)
with appet:
    appet = st.selectbox("Appetite", options=["Good", "Poor"], index=0)



# Prediction button
if st.button("Predict"):
    try:
        # Prepare input for the model
        input_features = [sg, al, sc, dm, htn, appet]
        prediction = predict(model, input_features)

        # Display the prediction
        st.success(f"The model prediction is: {prediction}")

        # Example visualization
        st.markdown("### Input Data Distribution")
        fig, ax = plt.subplots()
        ax.bar(["Feature 1", "Feature 2", "Feature 3"], input_features)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar for extra information
st.sidebar.title("About")
st.sidebar.write("This app predicts outputs based on the inputs provided using a trained model.")
