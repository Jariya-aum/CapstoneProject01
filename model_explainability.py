# model_explainability.py
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Load model
rf_model = joblib.load('models/best_model_RandomForest.pkl')

def plot_feature_importance(model, feature_names, top_n=15):
    """แสดง Feature Importance"""
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:][::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()


#plot_feature_importance(rf_model, X_train.columns)