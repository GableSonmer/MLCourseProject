import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, args):
        super(LSTMModel, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.output_size = args.O
        self.n_output = args.n_output
        self.dev = args.dev
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, 1)
        self.fcs = [nn.Linear(self.hidden_size, self.output_size, device=self.dev) for _ in range(self.n_output)]

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])
        preds = []
        for fc in self.fcs:
            preds.append(fc(out)[:, -1, :])
        out = torch.stack(preds, dim=0)
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size):
        super(TransformerModel, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=6)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out
