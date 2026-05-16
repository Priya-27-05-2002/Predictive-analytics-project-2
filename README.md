# 🎓 Student Performance Prediction and Early Intervention

## 👥 Team Members
- Umaparvathy C S  
- Lanka Priya  
- Shiha Shajahan  

---

## 📌 Problem Statement
Schools need a way to identify at-risk students early in the semester to provide counseling and academic support.  
This project aims to **predict whether a student will pass or fail a course** based on demographics, study habits, and attendance, allowing for timely intervention.

---

## 🏗️ Project Architecture
The project follows a modular and professional layout:

```bash
├── data
│   └── student_performance.csv
├── models
│   ├── best_model.pkl
│   └── preprocessor.pkl
│   └── LIME explanations
├── notebooks
│   └── 01_class_distribution.png
├── src
│   ├── data_preprocessing.py   # Handles missing values, scaling, encoding
│   ├── model_training.py       # Model definitions, hyperparameters, evaluation
│   ├── lime_analysis.py        # Model explainability scripts
│   ├── utils.py                # Helper functions
│   ├── app.py                  # Streamlit application
│   └── student_performance.py  # Main pipeline script
```


---

## 📊 Dataset Description
- **Source:** [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)  
- **Target Variable:** `Target_Pass`  
  - `1` → Final Grade ≥ 10 (Pass)  
  - `0` → Final Grade < 10 (Fail)  

---

## 🚀 Deployment
The project is deployed using **Streamlit**:  
👉 [Live Demo](https://predictive-analytics-2.streamlit.app/)
    
---

## ⚙️ Features
- Data preprocessing (handling missing values, scaling, encoding)  
- Model training with hyperparameter tuning  
- Model evaluation using metrics (accuracy, precision, recall, F1-score)  
- Explainability with **LIME**  
- Interactive **Streamlit app** for predictions
  ## 🛠️ How to Run Locally

Follow these three simple steps to set up and run the project on your machine:

### 1️. Clone the repository
```bash
git clone https://github.com/Priya-27-05-2002/Predictive-analytics-project-2.git
cd Predictive-analytics-project-2
```
### 2. Install dependencies
```bash
pip install -r requirements.txt

```
### 3. Run the Streamlit app
```bash
streamlit run src/app.py
```


---



