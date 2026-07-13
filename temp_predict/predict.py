import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

features = pd.read_csv('data.csv')
features.head()

print('The shape of our features is:', features.shape)

import datatime
years = features['Year']
months = features['Month']
days = features['Day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

features = pd.get_dummies(features)
features.head(5)

labels = np.array(features['actual'])
features= features.drop('actual', axis = 1)
feature_list = features.columns
features = np.array(features)

from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)




# build network model
x = torch.tensor(input_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float)

weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
biases = torch.randn(128, dtype=torch.float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)

learning_rate = 0.001
losses = []

for i in range(1000):
    # forward pass
    hidden_layer = torch.add(torch.mm(x, weights), biases)
    hidden_layer = torch.relu(hidden_layer)
    output_layer = torch.add(torch.mm(hidden_layer, weights2), biases2)

    # compute loss
    loss = torch.mean((output_layer - y.view(-1, 1)) ** 2)
    losses.append(loss.data.numpy())

    if i % 100 == 0:
        print(f'Epoch: {i} | Loss: {loss.data.numpy()}')

    # backward pass
    loss.backward()

    # update weights and biases
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        biases -= learning_rate * biases.grad
        weights2 -= learning_rate * weights2.grad
        biases2 -= learning_rate * biases2.grad

        # zero the gradients after updating
        weights.grad.zero_()
        biases.grad.zero_()
        weights2.grad.zero_()
        biases2.grad.zero_()