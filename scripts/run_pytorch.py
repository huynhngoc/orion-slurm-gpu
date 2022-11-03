from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len



if __name__ == '__main__':
    print("======================================================")
    print("Loading data")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    X = np.load('data/X_data.npy')
    y = np.load('data/y_data.npy')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    traindata = Data(X_train, Y_train)

    batch_size = 64
    trainloader = DataLoader(traindata, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    print("Finished loading data")
    print("======================================================")

    print("======================================================")
    print("Setting up neural network")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # number of features (len of X cols)
    input_dim = X.shape[1]
    # number of hidden layers
    hidden_layers = 25
    # number of classes (unique of y)
    output_dim = 1
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.linear1 = nn.Linear(input_dim, hidden_layers)
            self.linear2 = nn.Linear(hidden_layers, output_dim)

        def forward(self, x):
            x = torch.sigmoid(self.linear1(x))
            x = self.linear2(x)
            return x

    clf = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)

    print("Finished Setting up neural network")
    print("======================================================")


    print("======================================================")
    print("Training neural network")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')

    print("Finished Training neural network")
    print("======================================================")
