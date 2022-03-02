import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
N = 2000
# X = np.random.random(N)
X = torch.rand(N,1)
sign = (-torch.ones((N,1))) ** torch.randint(0,2,(N,1))
Y = np.sqrt(X) * sign


class NN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NN, self).__init__()
        self.i2h1 = nn.Linear(input_size, hidden_size1)
        self.h12h2 = nn.Linear(hidden_size1, hidden_size2)
        self.h2o = nn.Linear(hidden_size2, output_size)
        self.act = nn.Tanh()
        self.initialize_weights()

    def forward(self, input_tensor):
        # print(input_tensor.size())
        hidden1_i = self.i2h1(input_tensor)
        hidden1_o = self.act(hidden1_i)
        hidden2_i = self.h12h2(hidden1_o)
        hidden2_o = self.act(hidden2_i)
        output = self.h2o(hidden2_o)
        return output

    def initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


in_dim = 1
h1 = 10
h2 = 5
out_dim = 1

learner = NN(in_dim, h1, h2, out_dim)
criterion = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)


# def my_loss(output, target):
#     # loss = torch.mean((output - target)**2)
#     loss = torch.mean((target-output**2)**2)
#     return loss

def train_SGD(input_data, output_data):
    # input_data = torch.reshape(input_data,(in_dim,))
    # output_data = torch.reshape(output_data,(out_dim,))
    output = learner.forward(input_data)
    # loss = my_loss(output,output_data)
    loss = criterion(output**2,output_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


# def train_batch(inp,outp):
#     # inp = torch.tensor(inp)
#     # outp = torch.tensor(outp)
#     pred = learner.forward(inp)
#     loss = criterion(pred**2,outp)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return pred,loss




total_epochs = 100
for epoch in range(total_epochs):
    cum_loss = 0
    # for i in range(len(X)):
    # element_in = X[i]
    # true_o = Y[i]
    out, loss = train_SGD(X,X)
    print(loss)

X_test = torch.rand(N,1)
sign = (-torch.ones((N,1))) ** torch.randint(0,2,(N,1))
Y_test = np.sqrt(X_test) * sign

# result = np.zeros_like(X_test)
result = learner(X_test)
result = result.detach().numpy()
# for i in range(len(X_test)):
#     inp = torch.tensor(X_test[i],dtype=dtype)
#     out = learner(torch.reshape(inp,(1,1)))
#     result[i] = out.detach().cpu().numpy().reshape(out_dim, 1)

plt.figure(1)
plt.scatter(X_test, Y_test)
plt.scatter(X_test, result)
plt.show()
