import torch.nn.functional as F
import torch

loss_func = F.cross_entropy

def model(xb):
    return xb.mm(weights) + bias

bs = 64
xb = x_train[0:bs]
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bs = 64
bias = torch.zeros(10, requires_grad=True)
print(loss_func(model(xb), yb))


# build easier model
from torch import nn
class mnist_NN(nn.moudle):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 128)
        self.hiden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, xb):
        x = F.relu(self.lin1(xb))
        x = F.relu(self.hiden2(x))
        x = F.relu(self.out(x))
        return x

net = mnist_NN()
print(net)
