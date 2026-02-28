from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def randomForestTraining(X_train, y_train):
    print("Rozpoczynam Optymalizację Hiperparametrów dla Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)


    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_grid,
        n_iter=20,
        scoring='f1',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("\nOptymalizacja zakończona")
    print(f"Najlepsze parametry: {random_search.best_params_}")
    print(f"Najlepszy średni F1-Score z CV: {random_search.best_score_:.4f}")

    return random_search.best_estimator_



