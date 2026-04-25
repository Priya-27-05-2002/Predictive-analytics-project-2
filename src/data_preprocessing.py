import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(csv_path):
    print("Loading data from:", csv_path)
    df = pd.read_csv(csv_path)
    
    # Target definition (Stage 1 & 2)
    df['Target_Pass'] = (df['G3'] >= 10).astype(int)
    
    # Drop future grades
    df_clean = df.drop(columns=['G1', 'G2', 'G3'])
    
    X_model = df_clean.drop('Target_Pass', axis=1)
    y_model = df_clean['Target_Pass']
    
    numeric_features = X_model.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_model.select_dtypes(exclude=np.number).columns.tolist()
    
    # Preprocessing Pipeline (Stage 3)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.2, random_state=42, stratify=y_model
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_features, numeric_features
