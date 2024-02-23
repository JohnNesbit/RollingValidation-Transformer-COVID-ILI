import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn 
from torch import nn, Tensor
import torch.utils.data as data_utils
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFNN(nn.Module):
    def __init__(self,units_layer1,units_layer2,lags,horizons,n_features):
        super(FFNN, self).__init__()
        self.linear1 = nn.Linear(lags*n_features, units_layer1)
        self.linear2 = nn.Linear(units_layer1, units_layer2)
        self.linear3 = nn.Linear(units_layer2, horizons*n_features)  
        self.horizons = horizons
        self.n_features = n_features
    def forward(self, src, trg, train):
        src = torch.flatten(src, start_dim=1)
        src = torch.relu(self.linear1(src))
        src = torch.relu(self.linear2(src))
        output = self.linear3(src)
        output = torch.reshape(output, (output.shape[0],self.horizons, self.n_features)).to(device)
        return output[:,:,0]
    
    
class GRU(nn.Module):
    def __init__(self,hidden_size, num_layers,input_size, horizons,features):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizons)

    def forward(self, src, trg, train):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(device)
        # Forward propagate GRU
        out, _ = self.gru(src, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class LSTM(nn.Module):
    def __init__(self, hidden_dim, layer_dim, n_past,n_future,features):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.n_future = n_future
        self.lstm = nn.LSTM(features, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_future)
    def forward(self, src, trg, train):
        h0 = torch.zeros(self.layer_dim, src.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, src.size(0), self.hidden_dim).to(device)
        out, (hidden, cell) = self.lstm(src, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out