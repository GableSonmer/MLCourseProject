import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.n_input = 7
        self.n_output = 7
        self.device = configs.device

        self.hidden_size = 512
        self.num_layers = 2

        self.lstm = nn.LSTM(input_size=self.n_input,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=0.1,
                            batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size * self.seq_len, self.pred_len * self.n_output)

    def forward(self, x, y):
        batch_size = x.size(0)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h, c))
        out = self.norm(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = out.view(batch_size, self.pred_len, self.n_output)
        return out
