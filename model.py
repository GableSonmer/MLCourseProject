import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
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
