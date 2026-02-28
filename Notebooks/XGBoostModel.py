from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

def optimize_xgboost(X_train, y_train):
    print("Rozpoczynam Optymalizację Hiperparametrów (Random Search z Cross-Validation)...")
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8, 10, 12],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1.0, 2.15, 3.0]
    }

    xgb_base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=20,
        scoring='f1',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    print("\n Optymalizacja zakończona")
    print(f"Najlepsze parametry: {random_search.best_params_}")
    print(f"Najlepszy średni F1-Score z CV: {random_search.best_score_:.4f}")

    return random_search.best_estimator_

