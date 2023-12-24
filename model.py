"""
Models for the time series forecasting.

Input shape: (batch_size, seq_len[96], n_features[7])
Output shape: (batch_size, pred_len[336], n_output[7])
The [~] means the default value, in our dataset.
"""
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, seq_len, label_len, pred_len, device):
        super(LSTMModel, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.n_input = 7
        self.n_output = 7
        self.device = device

        self.hidden_size = 512
        self.num_layers = 2

        self.lstm = nn.LSTM(input_size=self.n_input, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size, self.pred_len) for _ in range(self.n_output)])

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        # (batch_size, seq_len, n_input)
        out, _ = self.lstm(x, (h, c))
        # (batch_size, seq_len, hidden_size)
        out = torch.stack([self.fcs[i](out)[:, -1, :] for i in range(self.n_output)], dim=2)
        # (batch_size, pred_len, n_output)
        return out


class TransformerModel(nn.Module):
    def __init__(self, seq_len, label_len, pred_len, device):
        super(TransformerModel, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.device = device
        self.n_input = 7
        self.n_output = 7

        # embedding layers for the input and output
        self.encoder = nn.Linear(self.n_input, 512)
        self.decoder = nn.Linear(self.n_output, 512)

        # Positional encoding
        # self.pos_encoder = nn.Parameter(torch.randn(seq_len, 1, 512))
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512))

        # Transformer
        self.transformer = nn.Transformer(d_model=512,
                                          nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          dim_feedforward=2048,
                                          dropout=0.1,
                                          batch_first=True)
        self.fc = nn.Linear(512, self.n_output)

    def forward(self, src, tgt):
        # embedding and positional encoding
        src = self.encoder(src) + self.positional_encoding[:src.size(1), :]
        tgt = self.decoder(tgt) + self.positional_encoding[:tgt.size(1), :]

        # transformer
        out = self.transformer(src, tgt)
        out = self.fc(out)
        # (batch_size, pred_len, n_output)
        return out

        # src = self.encoder(x)
        # src = src.permute(1, 0, 2)
        # src = src + self.pos_encoder
        # tgt = torch.zeros(x.size(0), self.pred_len, self.n_output).to(x.device)
        # tgt = self.decoder(tgt)
        # tgt = tgt.permute(1, 0, 2)
        # tgt = tgt + self.pos_encoder
        # out = self.transformer(src, tgt)
        # out = self.fc(out.permute(1, 0, 2))
        # return out
