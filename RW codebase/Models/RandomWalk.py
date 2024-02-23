class RandomWalk(nn.Module):
     def __init__(self,RWhorizon,seq_len,pred_len,enc_in ):
        super(RandomWalk, self).__init__()
        self.RWhorizon = RWhorizon
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.paramlist = nn.ParameterList()
        self.param = torch.nn.Parameter(torch.rand(1, 1, 4), requires_grad=True).to(device)
        self.paramlist.append(self.param)
    def forward(self, x, _, ___):
        batchSize = x.size(0)
        pred = x[:, :, :]
        #print(pred.shape)
        for walk in range(self.pred_len):
            #print(pred.shape)
            pred = torch.cat((pred, pred[:, walk-self.RWhorizon, :].unsqueeze(1) + self.param*.00001), 1)
        #print(pred.shape)
        #print(pred[:, -self.pred_len:, :].shape)
        return pred[:, -self.pred_len:, :][:, :, 0]