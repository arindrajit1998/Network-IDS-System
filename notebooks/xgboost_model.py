from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def run_xgb_and_evaluate(x_train, x_test, y_train, y_test):
    param_grid_xgb = {
        'n_estimators': [200],
        'max_depth': [12],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.5]
    }
    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        param_grid=param_grid_xgb,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid_search_xgb.fit(x_train, y_train)
    best_xgb_model = grid_search_xgb.best_estimator_
    y_pred_xgb = best_xgb_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred_xgb)
    precision = precision_score(y_test, y_pred_xgb, average='weighted')
    recall = recall_score(y_test, y_pred_xgb, average='weighted')
    f1 = f1_score(y_test, y_pred_xgb, average='weighted')

    summary = pd.DataFrame({
        'Model': ['XGBoost'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    return summary