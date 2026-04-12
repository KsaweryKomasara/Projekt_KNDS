import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def analyzeData(data):

    dataType = data.dtypes
    plt.rcParams['font.family'] = 'Arial'

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

def plotDistribution(data, var_type):
    df = data.copy()
    if var_type == 'cat':
        columns = df.select_dtypes(include=['object', 'category']).columns
        columns = columns.drop("Booking_ID", errors='ignore')
        plotCounts(df, columns)
    else : 
        columns = df.select_dtypes(include=['float64', 'int64']).columns
        notDiscreteValuesColumns = df[['lead_time', 'avg_price_per_room']]
        no_of_previous_bookings_not_canceled = df[['no_of_previous_bookings_not_canceled']]
        columns = columns.drop('lead_time', errors='ignore')
        columns = columns.drop('no_of_previous_bookings_not_canceled', errors='ignore')
        columns = columns.drop('avg_price_per_room', errors='ignore')
        plotCounts(df, columns)
        plotEcdf(df, no_of_previous_bookings_not_canceled)
        plotHist(df, notDiscreteValuesColumns)
        plotBoxPlots(df, notDiscreteValuesColumns)
        plotViolinPlots(df, notDiscreteValuesColumns)

def plotHist(data, columns):
    for column in columns:
        plt.figure(figsize=(12, 4))
        sns.set_style("darkgrid")
        sns.histplot(data[column], discrete= True, linewidth = 0.5, kde = True)
        ax = sns.histplot(data[column], discrete= True, linewidth = 0.5, kde = True)
        ## ax.set_yscale('log')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def plotCounts(data, columns):
    
    for column in columns:

        counts = data[column].value_counts()
        max_val = counts.max()
        min_val = counts.min()

        plt.figure(figsize=(12, 4))
        sns.set_style("darkgrid")
        sns.countplot(data=data, x=column)
        ax = sns.countplot(data=data, x=column) 
        ax.bar_label(ax.containers[0])
        if min_val > 0 and (max_val / min_val > 20):
            ax.set_yscale('log')
            plt.title(f'Rozkład {column}')
        else:
            plt.title(f'Rozkład {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def plotBoxPlots(data, columns):
    for column in columns:
        plt.figure(figsize=(12, 4))
        sns.set_style("darkgrid")
        sns.boxplot(data[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.show()

def plotViolinPlots(data, columns):
    for column in columns:
        plt.figure(figsize=(12, 4))
        sns.set_style("darkgrid")
        sns.violinplot(data[column])
        plt.title(f'Violin Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.show()

def plotEcdf(data, columns):
    for column in columns:
        plt.figure(figsize=(12, 4))
        sns.set_style("darkgrid")
        sns.ecdfplot(data[column])
        plt.title(f'ECDF of {column}')
        plt.xlabel(column)
        plt.ylabel('ECDF')
        plt.show()

def plotScatterDiagrams(data):
    columns = data.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            plt.figure(figsize=(12, 4))
            sns.set_style("darkgrid")
            sns.scatterplot(x=data[columns[i]], y=data[columns[j]])
            plt.title(f'Scatter Plot between {columns[i]} and {columns[j]}')
            plt.xlabel(columns[i])
            plt.ylabel(columns[j])
            plt.show()

def plotDiagramsForTargetVar(data, unique_val_threshold=10, dominance_threshold=0.8):
    target_column = data.columns[-1]
    print(f"Zmienna celu dla wykresów: {target_column}")
    
    numeric_columns = data.select_dtypes(include=['float64', 'int64'])
    
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        most_frequent_pct = data[column].value_counts(normalize=True).iloc[0]
        if data[column].nunique() <= unique_val_threshold:
            sns.countplot(data=data, x=column, hue=target_column)
            plt.title(f'Rozkład: {column} w zależności od {target_column}')
            plt.ylabel('Liczba wystąpień')
            
        elif most_frequent_pct > dominance_threshold:
            sns.stripplot(data=data, x=target_column, y=column, alpha=0.5, jitter=True)
            plt.title(f'Rozkład punktowy (dominanta {most_frequent_pct*100:.1f}%): {column} wg {target_column}')
        else:
            sns.boxplot(data=data, x=target_column, y=column)
            plt.title(f'Wykres pudełkowy: {column} wg {target_column}')
        plt.show()

def plotHeatmap(data): # Ta funkcja tworzy macierz korelacji dla zmiennych numerycznych i wizualizuje ją za pomocą heatmapy.
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Macierz korelacji')
    plt.show()

def plotComparisonWithTarget(data, var_type):
    df = data.copy()
    df.drop("Booking_ID", axis=1, inplace=True, errors='ignore')
    df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
    
    if var_type == 'cat':
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            plt.figure(figsize=(12, 4))
            sns.set_style("darkgrid")
            sns.countplot(data=df, x=column, hue='booking_status')
            ax = sns.countplot(data=df, x=column, hue='booking_status')
            ax.set_yscale('log')
            plt.title(f'Percentage distribution of {column} by booking_status')
            plt.xlabel(column)
            plt.ylabel('Percentage')
            plt.legend(title='booking_status', labels=['Not_Canceled', 'Canceled'])
            plt.show()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        notDiscreteValuesColumns = df[['lead_time', 'avg_price_per_room']]
        no_of_previous_bookings_not_canceled = df['no_of_previous_bookings_not_canceled']

        numeric_columns = numeric_columns.drop('lead_time', errors='ignore')
        numeric_columns = numeric_columns.drop('no_of_previous_bookings_not_canceled', errors='ignore')
        numeric_columns = numeric_columns.drop('avg_price_per_room', errors='ignore')

        ## countploty w dwóch słupkach na każdą wartosć numeryczną
        for column in numeric_columns:
            plt.figure(figsize=(12, 4))
            sns.set_style("darkgrid")
            sns.countplot(data=df, x=column, hue='booking_status')
            ax = sns.countplot(data=df, x=column, hue='booking_status')
            ax.set_yscale('log')
            plt.title(f'Count Plot of {column} by booking_status')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend(title='booking_status', labels=['Not_Canceled', 'Canceled'])
            plt.show()

        plt.figure(figsize=(12, 4))
        sns.set_style("darkgrid")
        sns.countplot(data=df, x=df['no_of_previous_bookings_not_canceled'].apply(lambda x: '0' if x == 0 else '>0'), hue='booking_status')
        ax = sns.countplot(data=df, x=df['no_of_previous_bookings_not_canceled'].apply(lambda x: '0' if x == 0 else '>0'), hue='booking_status')
        ax.set_yscale('log')
        plt.title(f'Count Plot of no_of_previous_bookings_not_canceled by booking_status')
        plt.xlabel('no_of_previous_bookings_not_canceled')
        plt.ylabel('Frequency')
        plt.legend(title='booking_status', labels=['Not_Canceled', 'Canceled'])
        plt.show()

        for column in notDiscreteValuesColumns:
            plt.figure(figsize=(12, 4))
            sns.set_style("darkgrid")
            sns.boxplot(x=df['booking_status'], y=df[column])
            plt.title(f'Box Plot: {column} vs booking_status')
            plt.xlabel('booking_status')
            plt.ylabel(column)
            plt.xticks([0, 1], ['Not_Canceled', 'Canceled'])
            plt.show()

def plotHeatmap(data):
    df = data.copy()
    df.drop("Booking_ID", axis=1, inplace=True, errors='ignore')
    df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Macierz korelacji dla zmiennych numerycznych')
    plt.show()

def plotCorrelationWithTarget(data):
    df = data.copy()
    df.drop("Booking_ID", axis=1, inplace=True, errors='ignore')
    df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix['booking_status'].to_frame(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Macierz korelacji zmiennych numerycznych ze zmienną celu: booking_status')
    plt.show()
