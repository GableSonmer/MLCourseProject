"""
Models for the time series forecasting.

Input shape: (batch_size, seq_len[96], n_features[7])
Output shape: (batch_size, n_output[7])
The [~] means the default value, in our dataset.
"""
import torch
import math
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.n_output = 7
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.n_output)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# class TransformerModel(nn.Module):
#     def __init__(self, input_size):
#         super(TransformerModel, self).__init__()
#         self.hidden_size = 64
#         self.num_layers = 2
#         self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=6)
#         self.fc = nn.Linear(input_size, 1)
#
#     def forward(self, x):
#         out = self.transformer(x)
#         out = self.fc(out[:, -1, :])
#         return out

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_output, d_model=256, nhead=8, num_encoder_layers=3, dim_feedforward=512,
                 dropout=0.1):
        super(TransformerModel, self).__init__()
        self.linear = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, n_output)

    def forward(self, src):
        src = self.linear(src)  # shape: [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # shape: [batch_size, seq_len, d_model]
        output = output.mean(dim=1)  # Aggregating across sequence
        output = self.output_layer(output)  # shape: [batch_size, n_output]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
