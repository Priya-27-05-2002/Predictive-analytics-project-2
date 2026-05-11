import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Early Intervention System", layout="centered")

st.title("🎓 Student Early Intervention System")
st.write("Predict if a student is at risk of failing based on their demographics and study habits.")

# Load the trained model from the 'models/' directory
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

try:
    model = load_model()
except:
    st.error("Model not found! Please run 'python student_performance.py' first.")
    st.stop()

# Input Form for Counselors
with st.form("student_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", ['F', 'M'])
        age = st.number_input("Age", min_value=10, max_value=25, value=15)
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], help="1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs")
        failures = st.number_input("Past Class Failures", min_value=0, max_value=4, value=0)
        absences = st.number_input("Absences", min_value=0, max_value=100, value=2)
        
    with col2:
        internet = st.selectbox("Internet Access at Home", ['yes', 'no'])
        higher = st.selectbox("Wants Higher Education", ['yes', 'no'])
        activities = st.selectbox("Extra-curricular Activities", ['yes', 'no'])
        health = st.slider("Current Health Status", 1, 5, 3)
        freetime = st.slider("Free Time After School", 1, 5, 3)
        
    # We will pass dummy values for the remaining features required by the dataset pipeline
    submit_button = st.form_submit_button("Predict Student Risk")

if submit_button:
    # Build base dictionary with default values for all dataset columns
    input_data = {
        'school': 'GP', 'sex': sex, 'age': age, 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
        'Medu': 4, 'Fedu': 4, 'Mjob': 'services', 'Fjob': 'services', 'reason': 'course',
        'guardian': 'mother', 'traveltime': 1, 'studytime': studytime, 'failures': failures,
        'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no', 'activities': activities, 'nursery': 'yes',
        'higher': higher, 'internet': internet, 'romantic': 'no', 'famrel': 4, 'freetime': freetime,
        'goout': 3, 'Dalc': 1, 'Walc': 1, 'health': health, 'absences': absences
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if prediction == 1:
        st.success(f"✅ **Safe Zone:** The student is predicted to PASS. (Confidence: {probability*100:.1f}%)")
    else:
        st.error(f"⚠️ **At-Risk:** The student is predicted to FAIL. Early intervention recommended. (Confidence: {(1-probability)*100:.1f}%)")
