# Exam Score Prediction

This project predicts a student's exam score from study habits, lifestyle factors, attendance, and background information. It includes exploratory data analysis, regression modeling, and simple deployment demos using FastAPI and Streamlit.

## Project Overview

The goal of this project is to estimate student exam performance using machine learning. The project starts in a Jupyter Notebook for analysis and modeling, then adds lightweight deployment files so the model can be tested through an API or an interactive web interface.

## Dataset

The dataset contains student-level features such as:

- Age
- Gender
- Study hours per day
- Social media hours
- Netflix hours
- Part-time job status
- Attendance percentage
- Sleep hours
- Diet quality
- Exercise frequency
- Parental education level
- Internet quality
- Mental health rating
- Extracurricular participation
- Exam score

## Workflow

1. Loaded and inspected the student habits dataset.
2. Checked duplicates and missing values.
3. Explored numerical and categorical feature distributions.
4. Split the data into features and target.
5. Built preprocessing pipelines for numerical and categorical columns.
6. Trained regression models and evaluated performance using R2 score.
7. Saved the final model as `best_model.pkl`.
8. Added FastAPI and Streamlit apps for simple deployment demos.

## Model Features

The deployed model uses these inputs:

- `age`
- `gender`
- `study_hours_per_day`
- `social_media_hours`
- `netflix_hours`
- `part_time_job`
- `attendance_percentage`
- `sleep_hours`
- `diet_quality`
- `exercise_frequency`
- `parental_education_level`
- `internet_quality`
- `mental_health_rating`
- `extracurricular_participation`

## Tech Stack

- Python
- pandas
- scikit-learn
- joblib
- FastAPI
- Streamlit
- Jupyter Notebook

## Project Structure

```text
.
+-- Exam Score Project.ipynb
+-- student_habits_performance.csv
+-- train_model.py
+-- best_model.pkl
+-- app.py
+-- streamlit_app.py
+-- requirements.txt
+-- README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python train_model.py
```

Run the FastAPI app:

```bash
uvicorn app:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## API Example

Send a `POST` request to `/predict`:

```json
{
  "age": 20,
  "gender": "Female",
  "study_hours_per_day": 3.5,
  "social_media_hours": 2.0,
  "netflix_hours": 1.0,
  "part_time_job": "No",
  "attendance_percentage": 90.0,
  "sleep_hours": 7.5,
  "diet_quality": "Good",
  "exercise_frequency": 3,
  "parental_education_level": "Bachelor",
  "internet_quality": "Good",
  "mental_health_rating": 8,
  "extracurricular_participation": "Yes"
}
```

Example response:

```json
{
  "predicted_exam_score": 82.45
}
```

## What I Learned

This project helped me practice the full regression workflow: exploratory data analysis, preprocessing, model training, evaluation, model saving, and creating simple deployment interfaces for portfolio demos.
