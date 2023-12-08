import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from model import LSTMModel, TransformerModel
from arguments import args
import joblib


def result_plot(args):
    dev = args.dev
    model = LSTMModel(7, args).to(args.dev)
    model.load_state_dict(torch.load('./saved_models/lstm_100.pth'))
    # load scaler for inverse transform
    scaler = joblib.load('scaler.pkl')

    data = pd.read_csv('data/ETTh1.csv', sep=',').drop(columns=['date'])
    start_index = np.random.randint(0, len(data) - 96 * 2 + 1)
    print(f'{start_index=}')
    data = data.iloc[start_index:start_index + 96 * 2, :]
    data = scaler.transform(data)

    predict_data = torch.tensor(data[:96], dtype=torch.float32).to(dev)
    labels = scaler.inverse_transform(data)

    date = range(96 * 2)

    model.eval()
    with torch.no_grad():
        for i in range(96):
            sequences = predict_data[i:96 + i].view(1, 96, 7)

            sequences = sequences.to(dev)

            outputs = model(sequences)

            predict_data = torch.cat([predict_data, outputs], dim=0)

        predict_data = scaler.inverse_transform(predict_data.cpu())

        for i in range(7):
            plt.subplot(7, 1, i + 1)
            plt.plot(date, predict_data[:, i].T, label=f'Prediction{i}')
            plt.plot(date, labels[:, i].T, label=f'Ground Truth{i}')
            plt.legend(loc='upper left', prop={'size': 4})
        plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result_plot(args)
