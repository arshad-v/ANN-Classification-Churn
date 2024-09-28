import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
import pickle

##Load trained model
model = tf.keras.models.load_model('model.h5')

## load the scaler and OHE
with open('ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streeamlit app
st.title("Customer Churn Prediction")

st.write("This app predicts the likelihood of a customer to churn based on their demographic and geographic information")

##User input

geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


##prepare Input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


## for geography
geo_encoder=ohe_geo.transform([[geography]]).toarray()
geo_encoder_df=pd.DataFrame(geo_encoder, columns=ohe_geo.get_feature_names_out(['Geography']))

##Combaain OHE with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

#### sacle
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')