
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import HfApi, hf_hub_download
import os

# Set page configuration
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

# Initialize Hugging Face API
api = HfApi()

# Define model repository ID and file name
repo_id_model = "hareeshkumarkb/tourism_prediction_model" # Replace with your actual model repo ID
model_filename = "best_tourism_prediction_model_v1.joblib"

@st.cache_resource
def load_model():
    """Downloads and loads the model from Hugging Face Hub."""
    try:
        # Download the model file from Hugging Face Hub
        model_path = hf_hub_download(repo_id=repo_id_model, filename=model_filename)
        model = joblib.load(model_path)
        st.success("Model loaded successfully from Hugging Face Hub.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

st.title("🌴 Tourism Package Purchase Predictor ✈️")
st.markdown("### This app will predict whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.")

st.sidebar.header("Customer Input Features")

# Function to get user input
def user_input_features():
    Age = st.sidebar.slider('Age', 18, 70, 30)
    TypeofContact = st.sidebar.selectbox('Type of Contact', ['Company Invited', 'Self Inquiry'])
    CityTier = st.sidebar.selectbox('City Tier', [1, 2, 3])
    DurationOfPitch = st.sidebar.slider('Duration of Pitch (minutes)', 1, 60, 15)
    Occupation = st.sidebar.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Freelancer'])
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Fe Male'])
    NumberOfPersonVisiting = st.sidebar.slider('Number of People Visiting', 1, 6, 2)
    NumberOfFollowups = st.sidebar.slider('Number of Follow-ups', 0, 6, 2)
    ProductPitched = st.sidebar.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    PreferredPropertyStar = st.sidebar.slider('Preferred Property Star Rating', 3, 5, 4)
    MaritalStatus = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
    NumberOfTrips = st.sidebar.slider('Number of Trips Annually', 1, 20, 5)
    Passport = st.sidebar.selectbox('Has Passport?', ['Yes', 'No'])
    OwnCar = st.sidebar.selectbox('Owns Car?', ['Yes', 'No'])
    NumberOfChildrenVisiting = st.sidebar.slider('Number of Children Visiting (below 5)', 0, 4, 0)
    Designation = st.sidebar.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])
    MonthlyIncome = st.sidebar.slider('Monthly Income', 10000, 100000, 30000)

    # Map categorical inputs to numerical/one-hot encoding format expected by the model's preprocessor
    # These mappings should correspond to how LabelEncoder and OneHotEncoder were fitted during training
    # For simplicity, using direct mapping for now. A more robust solution would be to use the fitted encoders.
    # Ensure these match the prep.py logic or the preprocessor in the pipeline.

    # Example mapping (adjust based on your actual LabelEncoder/OneHotEncoder steps)
    typeofcontact_map = {'Company Invited': 0, 'Self Inquiry': 1}
    occupation_map = {'Salaried': 2, 'Small Business': 3, 'Large Business': 1, 'Freelancer': 0} # Example mapping
    gender_map = {'Male': 1, 'Female': 0, 'Fe Male': 0} # Example mapping
    maritalstatus_map = {'Single': 2, 'Married': 1, 'Divorced': 0, 'Unmarried': 3} # Example mapping
    productpitched_map = {'Basic': 0, 'Deluxe': 1, 'Standard': 3, 'Super Deluxe': 4, 'King': 2} # Example mapping
    passport_map = {'Yes': 1, 'No': 0}
    owncar_map = {'Yes': 1, 'No': 0}
    designation_map = {'Executive': 1, 'Manager': 2, 'Senior Manager': 4, 'AVP': 0, 'VP': 5, 'Director': 3} # Example mapping

    data = {
        'Age': Age,
        'TypeofContact': typeofcontact_map[TypeofContact],
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': occupation_map[Occupation],
        'Gender': gender_map[Gender],
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': productpitched_map[ProductPitched],
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': maritalstatus_map[MaritalStatus],
        'NumberOfTrips': NumberOfTrips,
        'Passport': passport_map[Passport],
        'OwnCar': owncar_map[OwnCar],
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': designation_map[Designation],
        'MonthlyIncome': MonthlyIncome
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Customer Input')
st.write(input_df)

if st.button('Predict Purchase'):
    if model is not None:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.error(f"The customer is unlikely to purchase the package. (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.warning("Model is not loaded. Please check for errors in model loading.")

