# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") # Ignore potential warnings from sklearn/joblib

# Load the trained model and scaler
try:
    model = joblib.load('model.pkl') # Your trained SVC model from GridSearchCV
    scaler = joblib.load('scaler.pkl') # Your trained StandardScaler
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop execution if files are missing

# --- Streamlit UI ---

st.title("Telecom Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Input fields for the features used in the model
# Features were: "Age","Gender","MonthlyCharges","Tenure"

age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=130.0)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

# Convert gender input to numerical (Male=1, Female=0 as per your notebook)
gender_encoded = 1 if gender == "Male" else 0

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "MonthlyCharges": [monthly_charges],
    "Tenure": [tenure]
})

# Button to make prediction
if st.button("Predict Churn"):
    # Preprocess the input data using the loaded scaler
    # The scaler expects input as a DataFrame or array, and will output a numpy array
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input) # model is the GridSearchCV object
    
    st.balloons()

    # Interpret the prediction
    # Your Y column 'Churn' was encoded: Yes=1, No=0
    if prediction[0] == 1:
        result = "This customer is likely to Churn"
        st.error(result) # Use st.error for visual emphasis on churn
    else:
        result = "This customer is unlikely to Churn"
        st.success(result) # Use st.success for visual emphasis on no churn

    # Optional: Show the raw prediction result
    # st.write(f"Raw prediction: {prediction[0]}")