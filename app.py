from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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


class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Female", "Male", "Other"]
    study_hours_per_day: float = Field(..., ge=0)
    social_media_hours: float = Field(..., ge=0)
    netflix_hours: float = Field(..., ge=0)
    part_time_job: Literal["No", "Yes"]
    attendance_percentage: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=24)
    diet_quality: Literal["Poor", "Fair", "Good"]
    exercise_frequency: int = Field(..., ge=0, le=7)
    parental_education_level: Literal["None", "High School", "Bachelor", "Master"]
    internet_quality: Literal["Poor", "Average", "Good"]
    mental_health_rating: int = Field(..., ge=1, le=10)
    extracurricular_participation: Literal["No", "Yes"]


app = FastAPI(
    title="Exam Score Prediction API",
    description="A simple FastAPI demo for predicting student exam scores.",
    version="1.0.0",
)


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Run `python train_model.py` first.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Exam Score Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict")
def predict_exam_score(student: StudentData):
    model = load_model()
    input_data = pd.DataFrame(
        [
            {
                "age": student.age,
                "gender": student.gender,
                "study_hours_per_day": student.study_hours_per_day,
                "social_media_hours": student.social_media_hours,
                "netflix_hours": student.netflix_hours,
                "part_time_job": student.part_time_job,
                "attendance_percentage": student.attendance_percentage,
                "sleep_hours": student.sleep_hours,
                "diet_quality": student.diet_quality,
                "exercise_frequency": student.exercise_frequency,
                "parental_education_level": student.parental_education_level,
                "internet_quality": student.internet_quality,
                "mental_health_rating": student.mental_health_rating,
                "extracurricular_participation": student.extracurricular_participation,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(100.0, prediction))

    return {"predicted_exam_score": round(prediction, 2)}
