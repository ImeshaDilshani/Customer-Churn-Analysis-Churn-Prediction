import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.externals import joblib
import joblib


# Load the trained model
model = joblib.load('model.sav')

# # Function to preprocess input data
# def preprocess_input(data):
#     # Convert binary categorical variables to numerical
#     data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
#     data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
#     data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
#     data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
#     data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    
#     # Convert categorical variables to dummy variables
#     data = pd.get_dummies(data, columns=['MultipleLines', 'InternetService', 'Contract', 
#                                          'PaymentMethod', 'OnlineSecurity', 'OnlineBackup', 
#                                          'DeviceProtection', 'TechSupport', 'StreamingTV', 
#                                          'StreamingMovies'])
    
#     # Fill missing values
#     data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan).astype(float)
#     data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    
#     return data
# Function to preprocess input data
def preprocess_input(data):
    # List of columns to one-hot encode
    columns_to_encode = ['InternetService', 'Contract', 'PaymentMethod']
    
    # Convert binary categorical variables to numerical
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
    data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
    data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    
    # Convert categorical variables to dummy variables
    for column in columns_to_encode:
        if column in data.columns:
            data = pd.get_dummies(data, columns=[column])
    
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
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (months)', min_value=0, max_value=72, value=36)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 
                                                     'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=1000.0)

    # Create a dictionary with user inputs
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
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
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
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
