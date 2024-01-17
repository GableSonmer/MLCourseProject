"""
The oil data loader class.
@author: mysong
@date: 2023/12/22
@file: oil_dataset.py
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tools import StandardScaler


class OilDataset(Dataset):
    def __init__(self,
                 data_path='data/',
                 flag='train',
                 seq_len=96,
                 pred_len=96,
                 label_len=48,
                 scale=True,
                 inverse=True):
        assert flag in ['train', 'val', 'test']
        # data path
        self.data_path = data_path

        # init
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.scale = scale
        self.data_type = flag
        self.inverse = inverse

        # read data
        self._read_data()

    def _read_data(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path + self.data_type + '_set.csv')
        df_raw.drop(columns=['date'], inplace=True)

        # extract data
        cols_data = df_raw.columns[:]
        df_data = df_raw[cols_data]

        # scale if needed
        if self.scale:
            # use only train to scale, avoid data leakage
            train_data = pd.read_csv(self.data_path + 'train_set.csv')
            train_data = train_data.drop(columns=['date'])
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # get x and y
        self.data_x = data[:]
        if self.inverse:
            self.data_y = df_data.values[:]
        else:
            self.data_y = data[:]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = idx + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len + self.label_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # inverse scale
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len],
                                    self.data_y[r_begin + self.label_len:r_end]], 0)

        return seq_x, seq_y

    def inverse_transform(self, data):
        if self.scaler is None:
            raise Exception('Must set scale=True when init')
        return self.scaler.inverse_transform(data)
