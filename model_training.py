from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    print("Training models...")
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_auc = 0
    best_name = ""
    all_results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        all_results[name] = {
            "auc": auc, 
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"--- {name} ---")
        print(f"ROC AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model: {best_name} (AUC: {best_auc:.4f})")
    
    # Save artifacts (Stages 6 & 7)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(all_results, 'models/all_results.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    return best_name, best_auc
