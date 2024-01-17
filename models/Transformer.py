import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = configs.device
        self.n_input = 7
        self.n_output = 7

        # embedding layers for the input and output
        self.encoder = nn.Linear(self.n_input, 512)
        self.decoder = nn.Linear(self.n_output, 512)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512))

        # Transformer
        self.transformer = nn.Transformer(d_model=512,
                                          nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          dim_feedforward=2048,
                                          dropout=0.1,
                                          batch_first=True)

        # Output fully connected layer
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
