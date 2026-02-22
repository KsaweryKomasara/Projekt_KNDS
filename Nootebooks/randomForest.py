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

    print("inicializing RandomForestClassifier")
    rf = RandomForestClassifier()

    print("Training...")
    rf.fit(X_train,y_train)


    return rf


