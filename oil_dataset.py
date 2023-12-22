"""
The oil data loader class.
@author: mysong
@date: 2020/11/24
@file: oil_dataset.py
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tools import StandardScaler


class OilDataset(Dataset):
    def __init__(self,
                 data_path='data/ETTh1.csv',
                 flag='train',
                 seq_len=96,
                 pred_len=96,
                 label_len=48,
                 scale=True,
                 inverse=True):
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        # data path
        self.data_path = data_path

        # init
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.scale = scale
        self.set_type = type_map[flag]
        self.inverse = inverse

        # read data
        self._read_data()

    def _read_data(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        df_raw.drop(columns=['date'], inplace=True)

        # extract data
        cols_data = df_raw.columns[:]
        df_data = df_raw[cols_data]

        # scale if needed
        # use border1 and border2 to split train, val and test
        # first 12 months for train, second 4 months for val, last 4 months for test
        # total 20 months
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.scale:
            # use only train to scale, avoid data leakage
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # get x and y
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

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
