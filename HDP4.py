import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
import time

# Set page configuration with improved metadata
st.set_page_config(
    page_title="AI Health Assistant",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AI-powered Health Prediction System"
    }
)

# Adding enhanced custom CSS with animations and modern design
st.markdown("""
    <style>
        /* Modern gradient background with animation */
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Card styling for content */
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 2rem;
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
        }
        
        /* Modern title styling */
        .title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: -1px;
        }
        
        /* Subheading with animation */
        .subheading {
            font-size: 1.2rem;
            text-align: center;
            color: #6c757d;
            margin-bottom: 2rem;
            animation: fadeIn 2s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Enhanced input styling */
        .stNumberInput, .stSelectbox {
            border-radius: 10px !important;
            border: 1px solid #dee2e6 !important;
            transition: all 0.3s ease !important;
        }
        
        .stNumberInput:hover, .stSelectbox:hover {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Modern button styling */
        .stButton>button {
            border-radius: 10px !important;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 7px 14px rgba(0, 0, 0, 0.15) !important;
            opacity: 0.9 !important;
        }
        
        /* Result styling */
        .positive-result {
            background: linear-gradient(90deg, #ff4d4d 0%, #f94444 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 1rem;
            animation: pulse 2s infinite;
        }
        
        .negative-result {
            background: linear-gradient(90deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.9) !important;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .title {
                font-size: 2rem;
            }
            .subheading {
                font-size: 1rem;
            }
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Progress bar */
        .progress-container {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .progress-bar {
            height: 10px;
            border-radius: 10px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 1s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and resources
@st.cache_resource
def load_model():
    """Load the heart disease prediction model with caching"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), r'C:\Users\krish\Downloads\Copy_of_Project_10_Heart_Disease_Prediction.pickle')
        return pickle.load(open(model_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

heart_disease_model = load_model()

# Sidebar with enhanced navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2771/2771388.png", width=100)
    st.title("Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Heart Disease Prediction", "About", "Contact"],
        icons=["house", "heart-pulse", "info-circle", "envelope"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#667eea", "font-size": "16px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#e9ecef"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    # Add user profile section
    st.markdown("---")
    st.markdown("### User Profile")
    user_name = st.text_input("Your Name", placeholder="Enter your name")
    
    # Add health tips section
    st.markdown("---")
    st.markdown("### Daily Health Tip")
    tips = [
        "Drink at least 8 glasses of water daily",
        "Get 7-8 hours of sleep each night",
        "Walk at least 30 minutes every day",
        "Eat 5 servings of fruits and vegetables",
        "Practice stress-reduction techniques"
    ]
    st.info(np.random.choice(tips))

# Home Page
if selected == "Home":
    # Hero section with animation
    st.markdown('<h1 class="title">AI Health Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheading">Empowering your health decisions with artificial intelligence</p>', unsafe_allow_html=True)
    
    # Add a progress bar animation
    st.markdown('<div class="progress-container"><div class="progress-bar" id="progress"></div></div>', unsafe_allow_html=True)
    st.markdown("""
        <script>
            setTimeout(function() {
                document.getElementById('progress').style.width = '100%';
            }, 500);
        </script>
    """, unsafe_allow_html=True)
    
    # Features section with cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ❤️ Heart Health")
            st.markdown("Predict your risk of cardiovascular disease using advanced machine learning models.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Health Analytics")
            st.markdown("Visualize your health metrics and track improvements over time with interactive charts.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Personalized Insights")
            st.markdown("Receive customized health recommendations based on your unique profile and test results.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # How it works section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## How It Works")
        st.markdown("""
        1. **Select a health assessment** from the navigation menu
        2. **Enter your health information** in the provided form
        3. **Get instant AI-powered analysis** of your health status
        4. **Receive personalized recommendations** for improving your health
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Testimonials section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## What Our Users Say")
        
        testimonial_col1, testimonial_col2 = st.columns(2)
        
        with testimonial_col1:
            st.markdown("""
            > "This app helped me identify a potential heart issue early. The doctor confirmed the risk and I'm now taking preventive measures."
            
            *— Ram Patel., 52*
            """)
        
        with testimonial_col2:
            st.markdown("""
            > "As someone with a family history of heart disease, I find this tool incredibly valuable for regular check-ins."
            
            *— Jay Patel., 45*
            """)
        st.markdown("</div>", unsafe_allow_html=True)

# Heart Disease Prediction Page
elif selected == "Heart Disease Prediction":
    # Page header with animated title
    st.markdown('<h1 class="title">Heart Disease Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheading">Enter your health information below to assess your cardiovascular risk</p>', unsafe_allow_html=True)
    
    # Form with enhanced layout and tooltips
    with st.form(key='heart_disease_form'):
        # Create tabs for different sections of the form
        tab1, tab2, tab3 = st.tabs(["Basic Information", "Medical Metrics", "Exercise & ECG"])
        
        with tab1:
            st.markdown("### Personal Details")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input('Age (years)', min_value=1, max_value=120, value=45, 
                                    help="Enter your current age in years")
                sex = st.selectbox('Sex', ['Male', 'Female'], 
                                 help="Biological sex can influence heart disease risk")
                cp = st.selectbox('Chest Pain Type', 
                                ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], 
                                help="Type of chest pain experienced")
            
            with col2:
                trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120,
                                         help="Your blood pressure at rest")
                chol = st.number_input('Cholesterol Level (mg/dl)', min_value=100, max_value=600, value=200,
                                     help="Your serum cholesterol level")
        
        with tab2:
            st.markdown("### Blood & Health Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'],
                                  help="Is your fasting blood sugar above normal levels?")
                restecg = st.selectbox('Resting ECG Results', 
                                     ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
                                     help="Results from your resting electrocardiogram")
            
            with col2:
                thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=220, value=150,
                                        help="Your highest heart rate during exercise")
                exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'],
                                   help="Do you experience chest pain during exercise?")
        
        with tab3:
            st.markdown("### Exercise & Imaging Results")
            col1, col2 = st.columns(2)
            
            with col1:
                oldpeak = st.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                                  help="ST segment depression during exercise relative to rest")
                slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                                   ['Upsloping', 'Flat', 'Downsloping'],
                                   help="The slope of the ST segment during peak exercise")
            
            with col2:
                ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', 
                                 ['0', '1', '2', '3'],
                                 help="Number of major vessels visible on fluoroscopy")
                thal = st.selectbox('Thalassemia Test Result', 
                                  ['Normal', 'Fixed Defect', 'Reversible Defect'],
                                  help="Result of your thalassemia blood test")
        
        # Risk factors checklist
        with st.expander("Additional Risk Factors"):
            st.checkbox("Family history of heart disease")
            st.checkbox("Smoker or tobacco user")
            st.checkbox("Sedentary lifestyle")
            st.checkbox("High stress levels")
            st.checkbox("Diabetes diagnosis")
        
        # Form submission with enhanced button
        submitted = st.form_submit_button("Analyze My Heart Health", 
                                        help="Click to process your information and get your risk assessment")
    
    # Process form submission
    if submitted:
        if not heart_disease_model:
            st.error("Model not loaded. Please try again later.")
            st.stop()
        
        # Show processing animation
        with st.spinner("Analyzing your health data..."):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            time.sleep(0.5)
        
        # Convert inputs to model format
        sex_num = 1 if sex == 'Male' else 0
        cp_num = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp]
        fbs_num = 1 if fbs == 'Yes' else 0
        restecg_num = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[restecg]
        exang_num = 1 if exang == 'Yes' else 0
        slope_num = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[slope]
        thal_num = {'Normal': 2, 'Fixed Defect': 1, 'Reversible Defect': 0}[thal]
        
        # Create input DataFrame
        input_data = pd.DataFrame([[age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, 
                                   thalach, exang_num, oldpeak, slope_num, int(ca), thal_num]],
                                 columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Make prediction
        try:
            prediction = heart_disease_model.predict(input_data)
            probability = heart_disease_model.predict_proba(input_data)[0][1]
            
            # Display results with enhanced visualization
            st.markdown("## Results")
            
            if prediction[0] == 1:
                st.markdown(f'<div class="positive-result">⚠️ Potential Heart Disease Risk Detected</div>', 
                           unsafe_allow_html=True)
                st.warning(f"Risk Probability: {probability*100:.1f}%")
                
                # Show recommendations
                with st.expander("Recommended Actions"):
                    st.markdown("""
                    - **Consult a cardiologist** for further evaluation
                    - **Improve diet** with more fruits, vegetables, and whole grains
                    - **Increase physical activity** (aim for 150 min/week)
                    - **Monitor blood pressure** regularly
                    - **Reduce stress** through meditation or yoga
                    - **Quit smoking** if applicable
                    """)
                
                # Show risk factors
                st.markdown("### Contributing Risk Factors")
                risk_factors = []
                if age > 45: risk_factors.append(f"Age ({age})")
                if sex == 'Male': risk_factors.append("Male sex")
                if trestbps > 130: risk_factors.append(f"High blood pressure ({trestbps} mmHg)")
                if chol > 200: risk_factors.append(f"High cholesterol ({chol} mg/dl)")
                
                if risk_factors:
                    st.write("The following factors may be contributing to your risk:")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.info("No major traditional risk factors identified. Other factors may be influencing your risk.")
                
            else:
                st.markdown(f'<div class="negative-result">✅ Low Heart Disease Risk</div>', 
                           unsafe_allow_html=True)
                st.success(f"Risk Probability: {probability*100:.1f}%")
                
                # Show maintenance tips
                with st.expander("How to Maintain Heart Health"):
                    st.markdown("""
                    - **Continue regular exercise** (30 min most days)
                    - **Maintain healthy diet** with balanced nutrients
                    - **Get annual check-ups** to monitor health
                    - **Manage stress** through healthy outlets
                    - **Avoid smoking** and limit alcohol
                    """)
            
            # Add visualization
            st.markdown("### Risk Level Visualization")
            st.progress(int(probability * 100))
            
            # Add disclaimer
            st.markdown("""
            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 2rem;">
            <strong>Disclaimer:</strong> This assessment is not a substitute for professional medical advice. 
            Always consult with a qualified healthcare provider for personal health concerns.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# About Page
elif selected == "About":
    st.markdown('<h1 class="title">About Health Assistant</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ## Our Mission
        We believe in democratizing healthcare through artificial intelligence. Our goal is to make 
        advanced health risk assessments accessible to everyone, helping people make informed decisions 
        about their wellbeing.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ## Technology Stack
        - **Machine Learning**: Predictive models trained on clinical datasets
        - **Data Security**: All health data is processed locally in your browser
        - **Medical Validation**: Algorithms developed with input from cardiologists
        - **Continuous Improvement**: Models are regularly updated with new research
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ## The Team
        Our interdisciplinary team combines expertise in:
        - Artificial Intelligence & Machine Learning
        - Clinical Medicine & Cardiology
        - User Experience Design
        - Data Privacy & Security
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Contact Page
elif selected == "Contact":
    st.markdown('<h1 class="title">Contact Us</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Get In Touch
            Have questions or feedback? We'd love to hear from you!
            
            **Email**: kkpatel2020@gmail.com  
            **Phone**: +91 1234567891  
            **Address**: 94,Muktanand Society,GNFC Township, India
            """)
            
            # Contact form
            with st.form("contact_form"):
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
                message = st.text_area("Message")
                submitted = st.form_submit_button("Send Message")
                if submitted:
                    st.success("Thank you for your message! We'll respond within 48 hours.")
        
        with col2:
            # Map placeholder
            st.image("https://maps.googleapis.com/maps/api/staticmap?center=37.7749,-122.4194&zoom=13&size=600x400&maptype=roadmap", 
                    caption="Our Headquarters in Bharuch")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<footer style="text-align: center; padding: 1rem; margin-top: 2rem; color: #6c757d; font-size: 0.9rem;">
    © 2023 AI Health Assistant | Terms of Service | Privacy Policy
</footer>
""", unsafe_allow_html=True)