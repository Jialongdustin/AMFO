import torch.nn as nn


class BinaryNet(nn.Module):
    '''
    simple binary classification network with 1 hidden layer
    '''
    def __init__(self, input_dim, hidden_dim=64):
        super(BinaryNet, self).__init__()   
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    

        self.fc_layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_layer2 = nn.Linear(self.hidden_dim, 2)       
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        return x