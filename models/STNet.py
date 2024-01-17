import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # convolution part for extracting spatial features
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=3, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # residual part for extracting temporal features
        self.residual_conv = nn.Conv1d(in_channels=7, out_channels=64, kernel_size=1)

        # fully connected part for prediction
        self.fc = nn.Linear(in_features=64 * self.seq_len, out_features=7 * self.pred_len)

    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x + residual

        x = x.view(-1, 64 * self.seq_len)
        x = self.fc(x)
        x = x.view(-1, self.pred_len, 7)
        return x
