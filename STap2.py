import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import os

# Get the directory of the script
script_directory = os.path.dirname(__file__)

# Load the trained model
model = xgb.XGBRegressor()
model.load_model(os.path.join(script_directory, 'xgb_model.json'))

# Define the function for prediction
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app
st.title('XGBoost Model Prediction for the strength of cement-treated soils')

# Create two columns
col1, col2 = st.columns(2)

# Input fields for continuous features in the first column
with col1:
    LL = st.number_input('LL (Liquid Limit)', min_value=0.0, max_value=100.0, value=50.0)
    FC = st.number_input('FC (Fine Contents)', min_value=0.0, max_value=100.0, value=30.0)
    ρnorm = st.number_input('ρnorm (Normalized Dry Density)', min_value=0.5, max_value=1.5, value=0.8)
    ωnorm = st.number_input('ωnorm (Normalized Water Content)', min_value=0.5, max_value=1.5, value=0.9)

# Input fields for continuous features in the second column
with col2:
    C = st.number_input('C (Cement dosage)', min_value=0.0, max_value=10.0, value=6.0)
    T = st.number_input('T (Curing time)', min_value=0.0, max_value=90.0, value=28.0)
    nCiv = st.number_input('η/Civ (porosity to volumetric cement content ratio)', min_value=0.0, max_value=50.0, value=12.0)
    cem_type = st.selectbox('Select Cement Type', ['None', 'CEM_I', 'CEM_II', 'CEM_III'])

# Initialize CEM values
CEM_I = 0
CEM_II = 0
CEM_III = 0

# Set the selected CEM type
if cem_type == 'CEM_I':
    CEM_I = 1
elif cem_type == 'CEM_II':
    CEM_II = 1
elif cem_type == 'CEM_III':
    CEM_III = 1

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
    if cem_type == 'None':
        st.write('Please select a cement type to make a prediction.')
    else:
        prediction = predict(input_data)
        st.write(f'The predicted UCS value is: {prediction}')
