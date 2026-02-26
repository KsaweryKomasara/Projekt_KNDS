from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

def optimize_xgboost(X_train, y_train):
    print("Rozpoczynam Optymalizację Hiperparametrów (Random Search z Cross-Validation)...")
    
    # 1. Definiujemy przestrzeń poszukiwań (zakresy, z których algorytm będzie losował)
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8, 10, 12],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1.0, 2.15, 3.0] # Pozwalamy mu sprawdzić różne wagi dla klas
    }

    # 2. Inicjalizacja bazowego modelu
    xgb_base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

    # 3. Konfiguracja Random Search
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=20,          # Liczba losowych konfiguracji do sprawdzenia (możesz zwiększyć, jeśli masz czas)
        scoring='f1',       # Optymalizujemy pod kątem F1-Score (najważniejsze przy anulacjach)
        cv=5,               # K-Fold Cross-Validation: 5 foldów (części)
        verbose=2,          # Pokazuje postęp w konsoli
        random_state=42,
        n_jobs=-1
    )

    # 4. Uruchomienie wyszukiwania
    random_search.fit(X_train, y_train)

    print("\n✅ Optymalizacja zakończona!")
    print(f"Najlepsze parametry: {random_search.best_params_}")
    print(f"Najlepszy średni F1-Score z CV: {random_search.best_score_:.4f}")

    # Zwracamy najlepszy znaleziony model
    return random_search.best_estimator_

# WYWOŁANIE (podmień na swoje zmienne)
# best_xgb_model = optimize_xgboost(X_train_processed, y_train)