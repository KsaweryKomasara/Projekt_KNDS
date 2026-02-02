import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

def decisionTreeTrain(X_train, X_test, y_train, y_test):

    print("Training Decision Tree model...")

    model = DecisionTreeClassifier() # Inicjalizacja modelu drzewa decyzyjnego
    model.fit(X_train, y_train) # Trenowanie modelu na zbiorze treningowym

    print(f"Dataset 1 (Diagonal) - Decision Tree Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}") # Dokładność modelu drzewa decyzyjnego

    return model

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