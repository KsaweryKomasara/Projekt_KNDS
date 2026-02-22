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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

def adaBoostTraining(X_train, y_train):

    print("inicializing AdaBoostClassifier")
    ab = AdaBoostClassifier()

    print("Training...")
    ab.fit(X_train,y_train)


    return ab


def missclassificationError(y_true,y_pred):
    return 1-accuracy_score(y_true,y_pred)

def avaBoostConvergence(X_train,y_train,X_test,y_test):

    weak_learner=DecisionTreeClassifier(max_depth=1)
    dummy = DummyClassifier(strategy="most_frequent")
    n_estimators = 400

    weak_learners_misclassification_error = missclassificationError(
    y_test, weak_learner.fit(X_train, y_train).predict(X_test))

    dummy_classifiers_misclassification_error = missclassificationError(
    y_test, dummy.fit(X_train, y_train).predict(X_test))

    ad = AdaBoostClassifier(estimator=weak_learner, n_estimators=n_estimators)
    ad.fit(X_train,y_train)
    boosting_error = pd.DataFrame({
        "Trees": range(1, n_estimators + 1),
        "Ada": [missclassificationError(y_test, y_pred) for y_pred in ad.staged_predict(X_test)],

    })

    ax = boosting_error.plot(x="Trees", y="Ada", label="AdaBoost Error", figsize=(10, 6))
    plt.plot([boosting_error.index.min(), boosting_error.index.max()], [weak_learners_misclassification_error,weak_learners_misclassification_error],color="red", label="Weak Learner Error",linestyle="dotted")
    plt.plot([boosting_error.index.min(), boosting_error.index.max()], [dummy_classifiers_misclassification_error,dummy_classifiers_misclassification_error],color="green", label="Dummy Classifier Error", linestyle="dashed")
    plt.legend(["AdaBoost", "DecisionTreeClassifier", "DummyClassifier"], loc=1)
    plt.show()
    

