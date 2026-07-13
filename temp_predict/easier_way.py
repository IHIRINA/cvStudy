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


# build the model
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, output_size)
)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

losses = []
for epoch in range(1000):
    batch_losses = []
    for i in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        predictions = my_nn(xx)
        loss = cost(predictions, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_losses.append(loss.item())
        
    if epoch % 100 == 0:
        losses.append(np.mean(batch_losses))
        print('Epoch: {}, Loss: {}'.format(epoch, np.mean(batch_losses)))

x = torch.tensor(input_features, dtype=torch.float)
predictions = my_nn(x)
