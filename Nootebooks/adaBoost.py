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

def modelBenchmark(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy of the model:")
    print(classification_report(y_test,y_pred))

def FeaturesImportance(model,X_train):
    imp = model.feature_importances_
    ab_imp = pd.Series(imp, index=list(X_train.columns))
    std = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    #Coefficients of Variations
    cv = std/(imp+0.000001)
    cv = [t<0.25 for t in cv]

    fig, ax = plt.subplots()
    ab_imp.plot.bar(yerr=std, ax=ax)
    ax.set_title("Features Importance")
    ax.set_ylabel("mean impiurity deacrease")
    fig.tight_layout()
    print(cv)
    return cv

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
    plt.plot([boosting_error.index.min(), boosting_error.index.max()], [weak_learners_misclassification_error,weak_learners_misclassification_error],color="red", label="Weak Learner Error")
    plt.plot([boosting_error.index.min(), boosting_error.index.max()], [dummy_classifiers_misclassification_error,dummy_classifiers_misclassification_error],color="green", label="Dummy Classifier Error")
    plt.legend(["AdaBoost", "DecisionTreeClassifier", "DummyClassifier"], loc=1)
    plt.show()
    
def DecisionBoundryDisplay(ab,X,y):
    if hasattr(X,"values"):
        X=X.values
    if hasattr(y,"values"):
        y=y.values
    class_names = ["yes","no"]
    plot_colors = "br"
    plt.figure(figsize=(10, 6))
    ax=plt.subplot(121)
    disp=DecisionBoundaryDisplay.from_estimator(ab, X, cmap=plt.cm.Paired, response_method="predict", ax=ax,xlabel="x", ylabel="y")
    x_min, x_max = disp.xx0.min(), disp.xx0.max()
    y_min, y_max = disp.xx1.min(), disp.xx1.max()
    plt.axis("tight")
    for a,b,c in zip(range(2),class_names,plot_colors):
        idx = np.where(y == a)
        plt.scatter(X[idx, 0], X[idx, 1], c=c, edgecolor="black", s=20, label="Class %s" % c)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Decision Boundary of AdaBoost Classifier")
