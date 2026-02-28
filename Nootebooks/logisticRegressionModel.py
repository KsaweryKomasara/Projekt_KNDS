import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV

def logisticRegressionTrain(X_train, X_test, y_train, y_test):

    print("Training Logistic Regression model...")
    model = LogisticRegression() # Inicjalizacja modelu regresji logistycznej
    model.fit(X_train, y_train) # Trenowanie modelu na zbiorze treningowym

    print(f"Dataset 1 (Diagonal) - Logistic Regression Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}") # Dokładność modelu regresji logistycznej
    print("Coefficients of the Logistic Regression model:", model.coef_) # Współczynniki modelu
    print("Intercept of the Logistic Regression model:", model.intercept_) # Wyraz wolny modelu

    return model

def conductHPO(X_train, y_train):

    print("Conducting Hyperparameter Optimization for Logistic Regression...")

    param_grid = [
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs'],
            'penalty': ['l2'] # L2 nie jest obsługiwane przez solver 'lbfgs', dlatego ograniczamy się do L2 dla tego solvera
        },
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear'],
            'penalty': ["l1", "l2"]
        }
    ]

    grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters for Logistic Regression:", grid_search.best_params_)

    return grid_search.best_estimator_


def plot_decision_boundary(X, y, model, title): # To jest przydatna funckja do wizualizacji granic decyzyjnych modeli klasyfikacyjnych

    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto', alpha=0.6)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', edgecolor='white', s=40, label='Class 0 (True)')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', edgecolor='white', s=40, label='Class 1 (True)')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title, fontsize=15)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    plt.show()