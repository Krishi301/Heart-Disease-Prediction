import os
import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Adding custom CSS for the homepage design with a background image
st.markdown("""
    <style>
        /* Apply a background image */
        body {
            background-image: url('https://www.w3schools.com/w3images/medic.jpg'); /* Replace with a relevant health-related image URL */
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }

        /* Style the header */
        .title {
            color: #004d80;
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            margin-top: 80px;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.6);
        }

        /* Subheading text style */
        .subheading {
            font-size: 22px;
            text-align: center;
            margin-top: 20px;
            font-weight: 400;
            color: #f0f8ff;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
        }

        /* Add some padding and a shadow to the input form */
        .input-box {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            padding: 30px;
            margin: 20px 0;
        }

        /* Style the submit button */
        .stButton > button {
            background-color: #004d80;
            color: white;
            border-radius: 5px;
            padding: 15px 30px;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        /* Button hover effect */
        .stButton > button:hover {
            background-color: #003366;
            transform: scale(1.1);
        }

        /* Style the result section */
        .result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            color: #ffffff;
            background-color: #004d80;
            border-radius: 10px;
            margin-top: 20px;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .title {
                font-size: 40px;
            }
            .subheading {
                font-size: 18px;
            }
        }

        /* Heading Container */
        .header-container {
            text-align: center;
            margin-top: 30px;
        }

        /* Content Area */
        .content-container {
            max-width: 1200px;
            margin: auto;
            padding: 40px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved model
heart_disease_model = pickle.load(open(r'C:\Users\krish\Downloads\Copy_of_Project_10_Heart_Disease_Prediction.pickle', 'rb'))

# Sidebar Menu
selected = st.sidebar.selectbox("Select a Page", ["Home", "Heart Disease Prediction"])

# Home Page
if selected == 'Home':
    # Header with custom title and subtitle
    st.markdown('<div class="header-container"><h1 class="title">Welcome to Health Assistant 🧑‍⚕️</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="subheading">Your one-stop solution for health predictions and insights. Use AI to assess heart disease risks and more!</p>', unsafe_allow_html=True)
    
    # Main Content Area
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.write("### About This App")
    st.write(
        """
        This app uses machine learning models to predict health risks such as heart disease. By entering simple parameters like age, sex, and medical records, the model can help determine the likelihood of heart disease. 

        **Features:**
        - Heart Disease Prediction
        - Simple and easy-to-use interface
        - Real-time results
        - Data privacy considerations

        **How to use:**
        - Navigate to the **Heart Disease Prediction** page from the sidebar.
        - Input the required medical details.
        - Click on the button to get the prediction.

        We aim to assist healthcare professionals and individuals in identifying health risks early.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Page title with custom CSS class
    st.markdown('<h1 class="title">Heart Disease Prediction using ML 🫀</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheading">Use the form below to input the patient\'s information and predict the likelihood of heart disease.</p>', unsafe_allow_html=True)

    # Form for user input with custom CSS for input fields
    with st.form(key='heart_disease_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=30, step=1, key="age", help="Enter the patient's age.")
            sex = st.selectbox('Sex', ['Male', 'Female'], key="sex")
            cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], key="cp", help="Choose 0, 1, 2, or 3 based on chest pain type.")
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=120, step=1, key="trestbps")
        
        with col2:
            chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=100, max_value=600, value=200, step=1, key="chol")
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'], key="fbs")
            restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], key="restecg", help="Choose 0, 1, or 2")
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=220, value=150, step=1, key="thalach")
        
        with col3:
            exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'], key="exang")
            oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="oldpeak")
            slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2], key="slope", help="Choose 0, 1, or 2")
            ca = st.selectbox('Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3], key="ca", help="Choose number of vessels from 0 to 3")
            thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect'], key="thal")
        
        # Submit button with custom CSS class
        submit_button = st.form_submit_button(label='Heart Disease Test Result')

        if submit_button:
            # Convert user input to appropriate format for prediction
            sex = 1 if sex == 'Male' else 0
            fbs = 1 if fbs == 'Yes' else 0
            exang = 1 if exang == 'Yes' else 0
            thal = 2 if thal == 'Normal' else 1 if thal == 'Fixed Defect' else 0

            # Collect the inputs into a pandas DataFrame with the correct column names
            input_data = pd.DataFrame([[
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
            ]], columns=[
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ])

            # Predict using the model
            heart_prediction = heart_disease_model.predict(input_data)

            # Display result with custom CSS styling
            if heart_prediction[0] == 1:
                st.markdown('<div class="result">🔴 The person has a heart disease risk.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result">🟢 The person does not have any heart disease risk.</div>', unsafe_allow_html=True)

# End of script
