import lime
import lime.lime_tabular
import joblib

def run_lime_analysis(X_train, X_test, preprocessor, categorical_features, numeric_features):
    print("Running LIME analysis...")
    best_model = joblib.load('models/best_model.pkl')
    
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    X_train_dense = X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed
    X_test_dense = X_test_transformed.toarray() if hasattr(X_test_transformed, 'toarray') else X_test_transformed
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_dense,
        feature_names=all_feature_names,
        class_names=['Fail', 'Pass'],
        discretize_continuous=True
    )
    
    # Explain the first instance in the test set
    exp = explainer.explain_instance(
        X_test_dense[0],
        best_model.named_steps['classifier'].predict_proba,
        num_features=5
    )
    
    exp.save_to_file('models/lime_explanation.html')
    print("LIME explanation saved to models/lime_explanation.html")
