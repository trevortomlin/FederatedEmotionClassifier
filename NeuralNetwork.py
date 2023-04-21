import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(768, 6)
        
    def forward(self, x):
        x = self.fc1(x)
        return x