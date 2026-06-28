import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open("diabetes_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.sidebar.title("About")

st.sidebar.info("""
**Project:** Diabetes Prediction

**Model:** Support Vector Machine (Linear Kernel)

**Dataset:** PIMA Indians Diabetes Dataset

**Testing Accuracy:** 77.27%

Developed using:
- Python
- Scikit-learn
- Streamlit
""")

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's medical information below to predict "
    "whether they are likely to have diabetes."
)
with st.expander("📘 About this Project"):

    st.write("""
This application predicts the likelihood of diabetes using a
Support Vector Machine (SVM) trained on the PIMA Indians Diabetes Dataset.

### Features Used
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Note:** This tool is for educational purposes only and is not a substitute for professional medical advice.
""")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin", min_value=0.0, value=79.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=1, value=30)
if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        st.error("🔴 High Risk of Diabetes")
        st.warning("Please consult a healthcare professional.")
    else:
        st.success("🟢 Low Risk of Diabetes")

    st.info(f"Prediction Confidence: {confidence:.2f}%")


st.markdown("---")

st.caption(
    "Developed by Kartik Gupta | "
    "Machine Learning Project | "
    "InternPe"
)
st.caption("This application is for educational purposes only and should not be used as a medical diagnosis.")