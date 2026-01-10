import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dataAnalysis as dataAnalysis
import dataProcessing as dataProcessing

print("KNDS_Project")

def getData(filePath):
    dataset = pd.read_csv(dataFilePath)
    return dataset

def printData(data, rows):
    print(data.head(rows))


dataFilePath = 'Resources/Hotel_Reservations.csv'
data = getData(dataFilePath)

# Podstawowa analiza danych
#dataAnalysis.analyzeData(data)

# Analiza wartości odstających metodą kwartylową
# dataAnalysis.startIQRAnalysis(data)

# Zależności między zmiennymi
# Współczynnik korelacji Pearsona
# dataAnalysis.correlationCoefficient(data)
# dataAnalysis.plotScatterDiagram(data)

# Przetwarzanie danych
dataProcessing.processData(data)
