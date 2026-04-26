Student Performance Prediction and Early Intervention

Team Members :
Uma parvathy C.S
Lanka Priya
Shiha Shajahan
Problem Statement :
Schools need a way to identify at-risk students early in the semester to provide counseling and academic support. This project aims to predict whether a student will pass or fail a course based on demographics, study habits, and attendance, allowing for timely intervention.
Project Architecture
The project follows a highly modular, professional layout:
 data: Contains the dataset (student_performance.csv).
models: Stores trained model artifacts ( best_model.pkl,  preprocessor.pkl, LIME explanations).
notebooks : Contains generated plots (like 01_class_distribution.png).
src : Core Python modules for execution:
data_preprocessing.py : Handles missing values, scaling, and encoding.
model_training.py : Model definitions, hyperparameter setup, and evaluation metrics.
lime_analysis.py : Model explainability scripts.
 utils.py : Helper functions for directories.
 app.py : Streamlit application for deployment.
 student_performance.py : Main executable script tying the entire pipeline together.
 Dataset Description
Source: UCI Machine Learning Repository (Student Performance Dataset)
Target: Target_Pass  (Binary: 1 if Final Grade >= 10, else 0)


