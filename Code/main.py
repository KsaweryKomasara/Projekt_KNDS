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
# printData(data, 5)

dataAnalysis.analyzeData(data)

dataProcessing.processData(data)
