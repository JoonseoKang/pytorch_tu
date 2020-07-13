import torch

N, D_in, H, D_out = 64, 1000, 100, 10

class Twolayernet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(Twolayernet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)


    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

x.shape
y.shape

model = Twolayernet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 ==99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import random

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.output_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


model = DynamicNet(D_in, H, D_out)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()