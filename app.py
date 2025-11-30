import streamlit as st
import pandas as pd
import numpy as np
from ml_model import *  # Import your ML model functions

# Title
st.title("🏠 Housing Price Prediction")
st.write("Machine Learning Mini Model by Sandra Mkanyi")

# Sidebar for user input
st.sidebar.header("Input Features")

# Add input fields based on your model's features
# Example (adjust based on your actual features):
feature1 = st.sidebar.slider("Feature 1", 0, 100, 50)
feature2 = st.sidebar.slider("Feature 2", 0, 100, 50)
feature3 = st.sidebar.slider("Feature 3", 0, 100, 50)

# Predict button
if st.sidebar.button("Predict Price"):
    # Your prediction logic here
    st.success("Prediction will appear here!")
    
# Display your model information
st.header("📊 Model Information")
st.write("This model uses Linear Regression to predict housing prices.")