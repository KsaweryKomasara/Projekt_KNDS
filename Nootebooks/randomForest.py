from sklearn.ensemble import RandomForestClassifier


def randomForestTraining(X_train, y_train):

    print("inicializing RandomForestClassifier")
    rf = RandomForestClassifier()

    print("Training...")
    rf.fit(X_train,y_train)


    return rf
