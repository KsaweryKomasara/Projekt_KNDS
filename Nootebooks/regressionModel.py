import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def linearRegressionTrain(X_train_processed, X_test_processed, y_train, y_test):

    print("Training Linear Regression model...")

    model = LinearRegression(positive = False)

    model.fit(X_train_processed, y_train) # trenowanie modelu
    y_pred = model.predict(X_test_processed) # sprawdzenie na zbiorze testowym
    mse_train = mean_squared_error(y_train, model.predict(X_train_processed)) # ocena na zbiorze treningowym
    r2_train = r2_score(y_train, model.predict(X_train_processed)) # ocena na zbiorze treningowym
    mse_test = mean_squared_error(y_test, y_pred) # ocena na zbiorze testowym
    r2_test = r2_score(y_test, y_pred) # ocena na zbiorze testowym

    print(f"Linear Regression - Mean Squared Error, Train: {mse_train:.2f}, Test: {mse_test:.2f}")
    print(f"Linear Regression - R^2 Score, Train: {r2_train:.2f}, Test: {r2_test:.2f}")
    
def lassoRegressionTrain(X_train_processed, X_test_processed, y_train, y_test):

    print("Training Lasso Regression model...")

    model = Lasso(alpha=0.1, positive = True)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    mse_train = mean_squared_error(y_train, model.predict(X_train_processed))
    r2_train = r2_score(y_train, model.predict(X_train_processed))
    mse_test = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    print(f"Lasso Regression - Mean Squared Error, Train: {mse_train:.2f}, Test: {mse_test:.2f}")
    print(f"Lasso Regression - R^2 Score, Train: {r2_train:.2f}, Test: {r2_test:.2f}")

    return model

def ridgeRegressionTrain(X_train_processed, X_test_processed, y_train, y_test):

    print("Training Ridge Regression model...")

    model = Ridge(alpha=1.0, positive = True)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    mse_train = mean_squared_error(y_train, model.predict(X_train_processed))
    r2_train = r2_score(y_train, model.predict(X_train_processed))
    mse_test = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    print(f"Ridge Regression - Mean Squared Error, Train: {mse_train:.2f}, Test: {mse_test:.2f}")
    print(f"Ridge Regression - R^2 Score, Train: {r2_train:.2f}, Test: {r2_test:.2f}")

    return model