import matplotlib.pyplot as plt
import seaborn as sns


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

def plotHist(data):
    columns = data.select_dtypes(include=['float64', 'int64']).columns
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Rozkład zmiennej {column}')
        plt.xlabel(column)
        plt.ylabel('Liczba wystąpień')
        plt.show()

def plotCounts(data):
    columns = data.select_dtypes(include=['object', 'category']).columns
    columns = columns.drop("Booking_ID", errors='ignore')
    
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x=column)
        plt.title(f'Wykres liczności dla zmiennej {column}')
        plt.xlabel('Liczba wystąpień')
        plt.ylabel(column)
        plt.show()

def plotScatterDiagrams(data):
    columns = data.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[columns[i]], y=data[columns[j]])
            plt.title(f'Scatter Plot between {columns[i]} and {columns[j]}')
            plt.xlabel(columns[i])
            plt.ylabel(columns[j])
            plt.show()

def plotBoxDiagramsForTargetVar(data):
    target_column = data.columns[-1]
    print(f"Target variable for box plots: {target_column}")
    numeric_columns = data.select_dtypes(include=['float64', 'int64'])
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[target_column], y=data[column])
        plt.title(f'Box Plot of {column} by {target_column}')
        plt.xlabel(target_column)
        plt.ylabel(column)
        plt.show()