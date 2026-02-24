import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

def plotHeatmap(data): # Ta funkcja tworzy macierz korelacji dla zmiennych numerycznych i wizualizuje ją za pomocą heatmapy.
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Macierz korelacji')
    plt.show()

def plotCorrelationWithTarget(data):
    df = data.copy()
    df.drop("Booking_ID", axis=1, inplace=True, errors='ignore')
    df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix['booking_status'].to_frame(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Macierz korelacji ze zmienną celu: booking_status')
    plt.show()

def plotBarCharts(data):

    df = data.copy()

    columns = df.columns
    columns = columns.drop("Booking_ID", errors='ignore')
    columns = columns.drop("booking_status", errors='ignore')
    columns = columns.drop("arrival_date", errors='ignore')
    columns = columns.drop("no_of_previous_bookings_not_canceled", errors='ignore')
    columns = columns.drop("lead_time", errors='ignore')
    columns = columns.drop("avg_price_per_room", errors='ignore')


    counts = df.groupby(['no_of_children', 'booking_status']).size().unstack()
    cancel_rates = (counts['Canceled'] / counts.sum(axis=1) * 100).fillna(0)

    for columnName in columns:

        counts = df.groupby([columnName, 'booking_status']).size().unstack()
        cancel_rates = (counts['Canceled'] / counts.sum(axis=1) * 100).fillna(0)

        plt.figure(figsize=(16, 6))
        ax = sns.countplot(data=df, x=columnName, hue='booking_status')
        plt.title(f'Wykres liczności dla zmiennej {columnName} z podziałem na booking_status')
        plt.xlabel(columnName)
        plt.ylabel('Liczba wystąpień')
        plt.legend(title='Booking Status', loc='upper right')

        for container in ax.containers:
            ax.bar_label(container)

        for i, rate in enumerate(cancel_rates):
           max_height = max([p.get_height() for p in ax.patches if abs(p.get_x() + p.get_width()/2 - i) < 0.5])
           ax.text(i, max_height + 500, f'{rate:.1f}% cancel', ha='center', fontweight='bold', color='red')

        plt.show()

def plotDenseBarCharts(data):
    df = data.copy()
    df["lead_time_range"] = pd.cut(df["lead_time"], bins=[-1, 30, 60, 90, 120, float('inf')], labels=['0-30', '31-60', '61-90', '91-120', '120+'])
    df["avg_price_per_room_range"] = pd.cut(df["avg_price_per_room"], bins=[-1, 50, 100, 150, 200, float('inf')], labels=['0-50', '51-100', '101-150', '151-200', '200+'])

    columns = df[["lead_time_range", "avg_price_per_room_range"]]

    for columnName in columns:

        counts = df.groupby([columnName, 'booking_status']).size().unstack()
        cancel_rates = (counts['Canceled'] / counts.sum(axis=1) * 100).fillna(0)

        plt.figure(figsize=(12, 6))
        ax = sns.countplot(data=df, x=columnName, hue='booking_status')
        plt.title(f'Wykres liczności dla zmiennej {columnName} z podziałem na booking_status')
        plt.xlabel(columnName)
        plt.ylabel('Liczba wystąpień')
        plt.legend(title='Booking Status', loc='upper right')

        for container in ax.containers:
            ax.bar_label(container)

        for i, rate in enumerate(cancel_rates):
           max_height = max([p.get_height() for p in ax.patches if abs(p.get_x() + p.get_width()/2 - i) < 0.5])
           ax.text(i, max_height + 500, f'{rate:.1f}% cancel', ha='center', fontweight='bold', color='red')

        plt.show()