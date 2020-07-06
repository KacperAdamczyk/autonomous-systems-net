import csv
import neurolab as nl
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Parametry uczenia
hiddenLayer = 10
epochs = 1000
errorGoal = 0.01

# Import danych
headers = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
           'Magnesium', 'Total phenols', 'Flavanoids',
           'Nonflavanoid phenols', 'Proanthocyanins',
           'Color intensity', 'Hue',
           'OD280/OD315 of diluted wines', 'Proline']
wine_data = pd.read_csv('wine.data', sep=',', names=headers)

# Przygotowanie zbiorów
X = wine_data.drop(columns=['Class'])
y = wine_data['Class'].values.reshape(X.shape[0], 1)
# Normalizacja klas do przedziału [0; 1]
scaller = MinMaxScaler()
y = scaller.fit_transform(y)
# Podział danych na zbiory uczące i testujące
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=2)

# Utworzenie 2 warstwowej sieci neuronowej
inputLayers = []
for (columnName, columnData) in Xtrain.iteritems():
    inputLayers.append([min(columnData), max(columnData)])
net = nl.net.newff(inputLayers, [hiddenLayer, 1])

# Uczenie sieci
error = net.train(Xtrain, ytrain, epochs=epochs, goal=errorGoal, show=epochs)

# Symulacja wyjść sieci
outTrain = net.sim(Xtrain)
outTest = net.sim(Xtest)

# Prezentacja wyników
pl.subplot(311)
pl.plot(error)
pl.xlabel('Epoka')
pl.ylabel('Błąd SSE')
pl.subplot(312)
pl.xlabel('Klasa')
pl.ylabel('Numer wejścia')
pl.plot(outTrain)
pl.plot(ytrain)
pl.legend(['Wyjście dla zbioru uczącego', 'Oczekiwana wartość'])
pl.subplot(313)
pl.xlabel('Klasa')
pl.ylabel('Numer wejścia')
pl.plot(outTest)
pl.plot(ytest)
pl.legend(['Wyjście dla zbioru testowego', 'Oczekiwana wartość'])
pl.show()
