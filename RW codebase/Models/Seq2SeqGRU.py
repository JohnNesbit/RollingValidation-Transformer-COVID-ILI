import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn 
from torch import nn, Tensor
import torch.utils.data as data_utils
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUEncoder(nn.Module):
    def __init__(self, hidden_dim, layer_dim, lags, horizons, n_features):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(n_features, hidden_dim, layer_dim, batch_first=True)

    def forward(self, x):
        out, hidden = self.gru(x)
        return hidden
    
class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, layer_dim, lags, horizons, n_features):
        super(GRUDecoder, self).__init__()
        self.gru = nn.GRU(n_features, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.fc4 = nn.Linear(hidden_dim, n_features)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        out_1 = self.fc1(torch.squeeze(output))
        out_4 = self.fc4(hidden[-1])
        return out_1, out_4, hidden
    
class GRU_Seq2Seq(nn.Module):    
    def __init__(self,  hidden_dim, layer_dim, lags, horizons, n_features):
        super(GRU_Seq2Seq, self).__init__()
        self.encoder = GRUEncoder(hidden_dim, layer_dim, lags, horizons, n_features)
        self.decoder = GRUDecoder(hidden_dim, layer_dim, lags, horizons, n_features)
        self.horizons = horizons
        
    def forward(self, src, trg, train):
        hidden = self.encoder(src)
        outputs = torch.zeros(trg.shape[0], self.horizons).to(device)
        input_trg = src[:,src.shape[1]-1:src.shape[1],:]
        start = 0
        end = 1
        if(train == True):            
            for t in range(0, self.horizons):
                out_1, out_4, hidden = self.decoder(input_trg, hidden)
                outputs[:,start:end] = out_1
                if random.random() < 0.4:
                    input_trg = trg[:,start:end,:]
                else:
                    input_trg = out_4.unsqueeze(1)
                start = end 
                end = end + 1
        elif(train == False):
            for t in range(0, self.horizons):
                out_1, out_4, hidden = self.decoder(input_trg, hidden)
                outputs[:,start:end] = out_1
                input_trg = out_4.unsqueeze(1)
                start = end 
                end = end + 1
        return outputs