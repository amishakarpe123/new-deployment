import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# Page Configuration
st.set_page_config(page_title="Customer Purchase Predictor", layout="centered")

# Custom CSS for Animation and Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie Animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# Load the model
@st.cache_resource
def load_model():
    with open("model (4).pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header Section
st.title("🎯 Customer Ad Click Predictor")
st_lottie(lottie_coding, height=200, key="coding")
st.write("Enter the customer details below to predict the likelihood of a purchase.")

# Input Form
with st.container():
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
    
    with col2:
        salary = st.number_input("Estimated Salary ($)", min_value=0, value=50000, step=1000)

# Preprocessing Input
# Map Gender to numeric (Assuming 1 for Male, 0 for Female based on standard encoding)
gender_numeric = 1 if gender == "Male" else 0

# Prediction Logic
if st.button("Predict Outcome"):
    features = np.array([[gender_numeric, age, salary]])
    prediction = model.predict(features)
    
    st.divider()
    if prediction[0] == 1:
        st.balloons()
        st.success("### ✅ Result: The customer is likely to purchase!")
    else:
        st.info("### ❌ Result: The customer is unlikely to purchase.")
