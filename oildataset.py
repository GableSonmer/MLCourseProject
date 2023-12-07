import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

import config

np.random.seed(42)


def windowing(values, temps, seq_len):
    data = []
    labels = []
    n = len(values)
    for i in range(n - seq_len * 2):
        data.append(values[i:i + seq_len])
        labels.append(temps[i + seq_len:i + 2 * seq_len])
    return np.array(data), np.array(labels)


class OilDataset(Dataset):
    def __init__(self, seq_len=96, type='train'):
        filename = 'data/ETTh1.csv'
        df = pd.read_csv(filename, sep=',')
        df.drop(columns=['date'], inplace=True)
        df = df.astype('float32')
        # 归一化 和 反归一化,使用均值scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df)
        df = scaler.transform(df)

        # save scaler
        joblib.dump(scaler, 'scaler.pkl')

        # windowing
        X = df.tolist()
        y = X.copy()
        features, labels = windowing(X, y, seq_len)
        features = features.astype(np.float32)
        labels = labels.astype(np.float32)

        # shuffle
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        features = features[indices]
        labels = labels[indices]

        # size
        n = len(labels)

        # train test split, 6:2:2
        if type == 'train':
            self.features = features[:int(n * 0.6)]
            self.labels = labels[:int(n * 0.6)]
        elif type == 'val':
            self.features = features[int(n * 0.6):int(n * 0.8)]
            self.labels = labels[int(n * 0.6):int(n * 0.8)]
        elif type == 'test':
            self.features = features[int(n * 0.8):]
            self.labels = labels[int(n * 0.8):]
        else:
            raise ValueError('type must be train, val or test')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
