def analyzeData(data):
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())