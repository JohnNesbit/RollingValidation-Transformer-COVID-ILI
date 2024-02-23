import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.hidden_dim = configs.d_model
        self.layer_dim = configs.layer_dim
        self.n_future = configs.pred_len
        self.enc_in = configs.enc_in
        self.features = configs.enc_in
        self.lstm = nn.LSTM(configs.enc_in, configs.d_model, configs.layer_dim, batch_first=True)
        self.fc = nn.Linear(configs.d_model, configs.pred_len*configs.enc_in)
    def forward(self, src, trg, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=False):
        h0 = torch.zeros(self.layer_dim, src.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, src.size(0), self.hidden_dim).to(device)
        out, (hidden, cell) = self.lstm(src, (h0, c0))
        out = self.fc(out[:, -1, :]).reshape(32, self.n_future, self.enc_in)
        #print(out.shape)
        return out