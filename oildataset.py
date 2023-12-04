import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)


def windowing(values, temps, seq_len):
    data = []
    labels = []
    n = len(values)
    for i in range(n - seq_len):
        data.append(values[i:i + seq_len])
        labels.append(temps[i + seq_len])
    return np.array(data), np.array(labels)


class OilDataset(Dataset):
    def __init__(self, seq_len=96, type='train'):
        filename = 'data/ETTh1.csv'
        df = pd.read_csv(filename, sep=',')
        df.drop(columns=['date'], inplace=True)
        df = df.astype('float32')

        # windowing
        X = df.drop(columns=['OT']).values
        y = df['OT'].values
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
