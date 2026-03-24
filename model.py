import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NeuralNet, self).__init__()

        self.fc1     = nn.Linear(input_size, hidden1_size)
        self.fc2     = nn.Linear(hidden1_size, hidden2_size)
        self.fc3     = nn.Linear(hidden2_size, output_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x