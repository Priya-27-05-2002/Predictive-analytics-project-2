import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import create_directories
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_and_evaluate
from src.lime_analysis import run_lime_analysis

if __name__ == "__main__":
    # 1. Setup Architecture
    create_directories()
    
    # 2. Data Loading & Preprocessing
    csv_path = 'data/student_performance.csv'
    X_train, X_test, y_train, y_test, preprocessor, cat_features, num_features = load_and_preprocess_data(csv_path)
    
    # 3. Exploratory Data Analysis (EDA) - Save to notebooks/
    df = pd.read_csv(csv_path)
    df['Target_Pass'] = (df['G3'] >= 10).astype(int)
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Target_Pass', hue='Target_Pass', palette="Set2", legend=False)
    plt.title("Class Distribution (0 = Fail, 1 = Pass)")
    plt.savefig("notebooks/01_class_distribution.png")
    plt.close()
    print("EDA Plot saved to notebooks/01_class_distribution.png")
    
    # 4. Model Training & Evaluation
    best_name, best_auc = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    # 5. Model Interpretability
    run_lime_analysis(X_train, X_test, preprocessor, cat_features, num_features)
    
    print("\n✅ Full pipeline completed successfully!")
