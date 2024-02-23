import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self,configs):
        super(GRU, self).__init__()
        self.horizons = configs.pred_len
        self.features = configs.enc_in
        self.hidden_size = configs.d_model
        self.enc_in = configs.enc_in
        self.num_layers = configs.layer_dim
        self.gru = nn.GRU(configs.enc_in, configs.d_model, configs.layer_dim, batch_first=True)
        self.fc = nn.Linear(configs.d_model, configs.pred_len*configs.enc_in)

    def forward(self, src, trg, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=False):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(device)
        # Forward propagate GRU
        out, _ = self.gru(src, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]).reshape(32, self.horizons, self.enc_in)
        return out