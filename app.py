import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('model.h5')
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Load label encoders
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geography.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f) 

st.title("Customer Churn Prediction")
# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", options=['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict Churn"):
    # Prepare input data
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    input_df = pd.DataFrame([input_data])
    # Encode categorical variables
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
    geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)
    input_df.drop('Geography', axis=1, inplace=True)
    # Scale the input data
    input_data_scaled = scaler.transform(input_df)
    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]
    if prediction_prob > 0.5:
        st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
    else:
        st.write(f"The customer is unlikely to churn with a probability of {prediction_prob:.2f}") 
# --- IGNORE ---