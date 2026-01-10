def analyzeData(data):

    dataType = data.dtypes

    print("Analyzing data...")
    print("Data Overview:")
    print("Data info: \n" + str(data.info()) + "\n")
    print("Data description: \n" + str(data.describe()) + "\n")
    print("Duplicated rows: \n" + str(data.duplicated().sum()) + "\n")
    print("Number of null values: \n" + str(data.isnull().sum()) + "\n")
    print("Data Types: \n" + str(dataType))

# Metoda kwartylowa

def startIQRAnalysis(data):

    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        # Wartości odstające
        outliers = data[(data[column] < lowerBound) | (data[column] > upperBound)]
        print(f"Number of outliers in {column}:\n", outliers.shape[0])

def correlationCoefficient(data):

    correlatioVector = []
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            corr = data[numeric_columns[i]].corr(data[numeric_columns[j]])
            correlatioVector.append([numeric_columns[i], numeric_columns[j], corr])

    for item in correlatioVector:
        print(item)

def plotScatterDiagram(data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[numeric_columns[i]], y=data[numeric_columns[j]])
            plt.title(f'Scatter Plot between {numeric_columns[i]} and {numeric_columns[j]}')
            plt.xlabel(numeric_columns[i])
            plt.ylabel(numeric_columns[j])
            plt.show()

