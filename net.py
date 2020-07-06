import csv
import neurolab as nl
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Parametry uczenia
hiddenLayers = [12, 7]
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
    X, y, test_size=0.2, random_state=42)

# Utworzenie sieci neuronowej
inputLayers = []
for (columnName, columnData) in Xtrain.iteritems():
    inputLayers.append([min(columnData), max(columnData)])
net = nl.net.newff(inputLayers, [*hiddenLayers, 1])

# Uczenie sieci
error = net.train(Xtrain, ytrain, epochs=epochs, show=epochs)

# Symulacja wyjść sieci
outTrain = net.sim(Xtrain)
outTest = net.sim(Xtest)

# Sortowanie tablic wynikowych dla bardziej przejrzystej prezentacji
train = []
test = []
for x, y in zip(outTrain, ytrain):
    train.append([*x, *y])
for x, y in zip(outTest, ytest):
    test.append([*x, *y])
train.sort(key=lambda pair: pair[1])
test.sort(key=lambda pair: pair[1])

# Prezentacja wyników
epochsElapsed = len(error)
print(f'Epochs: {len(error)}')
print(f'Error: {error}')
pl.figure(
    f'Max epochs = {epochs} Hidden layers = {len(hiddenLayers)} Neurons = {hiddenLayers}')
pl.subplot(311)
pl.title(
    f'Epoki: {epochsElapsed} Ukryte warstwy = {len(hiddenLayers)} Neurony = {hiddenLayers}')
pl.plot(error)
pl.xlabel('Epoka')
pl.ylabel('Błąd SSE')
pl.subplot(312)
pl.yticks([0, 0.5, 1])
pl.xlabel('Numer wejścia')
pl.ylabel('Klasa')
pl.plot(train)
pl.legend(['Wyjście dla zbioru uczącego', 'Oczekiwana wartość'])
pl.subplot(313)
pl.yticks([0, 0.5, 1])
pl.xlabel('Numer wejścia')
pl.ylabel('Klasa')
pl.plot(test)
pl.legend(['Wyjście dla zbioru testowego', 'Oczekiwana wartość'])
pl.show()
