import torch

# tensor格式

x = torch.empty(2, 3, requires_grad=True)

x = torch.rand(2, 3)

x = torch.zeros(2, 3)

x.size()

y = x.numpy()





# 神经网络
import torch
import torch.nn as nn

input_size = 1
output_size = 3


class net(nn.moudle):
    def __init__(self):
        super(net, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = net()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epoches = 100
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epoches):
    epoch += 1
    inputs = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    
    loss.backward()
    
    optimizer.step()   
    
    if epoch % 5 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        torch.save(model.state_dict(), 'model.pkl')
        # model.load_state_dict(torch.load('model.pkl'))
    