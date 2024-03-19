import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.externals import joblib
import joblib

# Load the trained model
model = joblib.load('model.sav')

# Function to preprocess input data
def preprocess_input(data):
    # Convert categorical variables to dummy variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
    data = pd.get_dummies(data, columns=categorical_columns)
    
    # Fill missing values
    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan).astype(float)
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    
    return data

# Define the layout of the app
def main():
    st.title('Customer Churn Prediction')

    # Add some description or instructions
    st.write('Enter the customer details below to predict churn.')

    # Add input fields for user input
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=1000.0)
    gender = st.selectbox('Gender', ['Female', 'Male'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'No phone service', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'No internet service', 'Yes'])
    online_backup = st.selectbox('Online Backup', ['No', 'No internet service', 'Yes'])
    device_protection = st.selectbox('Device Protection', ['No', 'No internet service', 'Yes'])
    tech_support = st.selectbox('Tech Support', ['No', 'No internet service', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'No internet service', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'No internet service', 'Yes'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', ['Bank transfer (automatic)', 
                                                     'Credit card (automatic)', 
                                                     'Electronic check', 
                                                     'Mailed check'])
    tenure = st.slider('Tenure (months)', min_value=1, max_value=72, value=36)

    # Create a dictionary with user inputs
    input_data = {
        'SeniorCitizen': senior_citizen,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'tenure': tenure
    }

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input data
    input_df = preprocess_input(input_df)

    # Add a button to make predictions
    if st.button('Predict'):
        # Make predictions using the model
        prediction = model.predict(input_df)
        # Display the prediction
        if prediction[0] == 0:
            st.write('Prediction: Not Churned')
        else:
            st.write('Prediction: Churned')

# Run the app
if __name__ == '__main__':
    main()

    
