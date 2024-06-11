import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = xgb.XGBRegressor()
model.load_model(r'C:\Users\khan4\Desktop\Python practice\0. Meetings\6. June 20, 2024\xgb_model.json')

# Define the function for prediction
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app
st.title('XGBoost Model Prediction for the strength of cement-treated soils')

# Input fields for continuous features
LL = st.number_input('LL (Liquid Limit)', min_value=0.0, max_value=100.0, value=50.0)
FC = st.number_input('FC (Fines Content)', min_value=0.0, max_value=100.0, value=30.0)
ρnorm = st.number_input('ρnorm (Normalized Dry Density)', min_value=0.5, max_value=1.5, value=0.8)
ωnorm = st.number_input('ωnorm (Normalized Water Content)', min_value=0.5, max_value=1.5, value=0.9)
C = st.number_input('C (Cement dosage)', min_value=0.0, max_value=10.0, value=6.0)
T = st.number_input('T (Curing time)', min_value=0.0, max_value=90.0, value=28.0)
nCiv = st.number_input('nCiv (porosity-C ratio)', min_value=0.0, max_value=50.0, value=12.0)

# Input fields for binary features
CEM_I = st.selectbox('CEM_I', [0, 1])
CEM_II = st.selectbox('CEM_II', [0, 1])
CEM_III = st.selectbox('CEM_III', [0, 1])

# Prepare input data
input_data = {
    'LL': LL,
    'FC': FC,
    'CEM_I': CEM_I,
    'CEM_II': CEM_II,
    'CEM_III': CEM_III,
    'ρnorm': ρnorm,
    'ωnorm': ωnorm,
    'C': C,
    'T': T,
    'nCiv': nCiv,

}

# Make prediction when button is clicked
if st.button('Predict'):
    prediction = predict(input_data)
    st.write(f'The predicted UCS value is: {prediction}')