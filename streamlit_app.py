from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"
FEATURE_COLUMNS = [
    "age",
    "gender",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Exam Score Prediction")

st.title("Exam Score Prediction")
st.write("Enter student habits and background details to estimate the exam score.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    study_hours_per_day = st.number_input(
        "Study hours per day",
        min_value=0.0,
        value=3.0,
        step=0.1,
    )
    social_media_hours = st.number_input(
        "Social media hours",
        min_value=0.0,
        value=2.0,
        step=0.1,
    )
    netflix_hours = st.number_input("Netflix hours", min_value=0.0, value=1.0, step=0.1)
    part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
    attendance_percentage = st.slider("Attendance percentage", 0.0, 100.0, 85.0)
    sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0)
    diet_quality = st.selectbox("Diet quality", ["Poor", "Fair", "Good"])
    exercise_frequency = st.slider("Exercise frequency per week", 0, 7, 3)
    parental_education_level = st.selectbox(
        "Parental education level",
        ["None", "High School", "Bachelor", "Master"],
    )
    internet_quality = st.selectbox("Internet quality", ["Poor", "Average", "Good"])
    mental_health_rating = st.slider("Mental health rating", 1, 10, 7)
    extracurricular_participation = st.selectbox(
        "Extracurricular participation",
        ["No", "Yes"],
    )
    submitted = st.form_submit_button("Predict score")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "study_hours_per_day": study_hours_per_day,
                "social_media_hours": social_media_hours,
                "netflix_hours": netflix_hours,
                "part_time_job": part_time_job,
                "attendance_percentage": attendance_percentage,
                "sleep_hours": sleep_hours,
                "diet_quality": diet_quality,
                "exercise_frequency": exercise_frequency,
                "parental_education_level": parental_education_level,
                "internet_quality": internet_quality,
                "mental_health_rating": mental_health_rating,
                "extracurricular_participation": extracurricular_participation,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(100.0, prediction))
    st.metric("Predicted exam score", f"{prediction:.2f}")
