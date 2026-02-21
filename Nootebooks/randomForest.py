import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def randomForestTraining(X_train, y_train):

    print("inicializing RandomForestClassifier");
    rf = RandomForestClassifier();

    print("Training...")
    rf.fit(X_train,y_train)


    return rf

def modelBenchmark(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy of the model:")
    print(classification_report(y_test,y_pred))

def FeaturesImportance(model,X_train):
    imp = model.feature_importances_
    forest_imp = pd.Series(imp, index=list(X_train.columns))
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    #Coefficients of Variations
    cv = std/imp
    cv = [t<0.25 for t in cv]

    fig, ax = plt.subplots()
    forest_imp.plot.bar(yerr=std, ax=ax)
    ax.set_title("Features Importance")
    ax.set_ylabel("mean impiurity deacrease")
    fig.tight_layout()
    print(cv)
    return cv

def confiusionMatrix(model,X_test,y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Reds)
    plt.title('Macierz pomyłek')
    plt.show()
