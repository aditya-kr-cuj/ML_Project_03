import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load the pre-trained model
model_path = 'house_price_model.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Please ensure the model is saved in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the feature scaler (reusing same preprocessing steps as training)
scaler = StandardScaler()

# Streamlit App Title
st.title("House Price Prediction App")

# Input fields for house features
st.header("Enter House Features")
mainroad = st.radio("Main Road Access:", ["Yes", "No"])
guestroom = st.radio("Guest Room Available:", ["Yes", "No"])
basement = st.radio("Basement Available:", ["Yes", "No"])
hotwaterheating = st.radio("Hot Water Heating Available:", ["Yes", "No"])
airconditioning = st.radio("Air Conditioning Available:", ["Yes", "No"])
prefarea = st.radio("Preferred Area:", ["Yes", "No"])
furnishingstatus = st.selectbox("Furnishing Status:", ["Semi-furnished", "Furnished", "Unfurnished"])

area = st.number_input("Enter the Area (sq ft):", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms:", min_value=1, max_value=5, value=2)
stories = st.number_input("Number of Stories:", min_value=1, max_value=5, value=1)
parking = st.number_input("Number of Parking Spaces:", min_value=0, max_value=5, value=1)

# Process user inputs into feature array
feature_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": 1 if mainroad == "Yes" else 0,
    "guestroom": 1 if guestroom == "Yes" else 0,
    "basement": 1 if basement == "Yes" else 0,
    "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
    "airconditioning": 1 if airconditioning == "Yes" else 0,
    "parking": parking,
    "prefarea": 1 if prefarea == "Yes" else 0,
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "Semi-furnished" else 0,
    "furnishingstatus_unfurnished": 1 if furnishingstatus == "Unfurnished" else 0,
}

# Convert to DataFrame
feature_df = pd.DataFrame([feature_dict])

# Scale the numerical features
try:
    # Load your scaler (if already saved during training, else comment this and use a manual scaler)
    # scaler = joblib.load("scaler.pkl")
    numeric_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    feature_df[numeric_columns] = scaler.fit_transform(feature_df[numeric_columns])
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(feature_df)
        st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
